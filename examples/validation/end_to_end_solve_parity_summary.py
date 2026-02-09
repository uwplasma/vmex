"""End-to-end solve parity snapshot (input -> solve -> compare).

This script is intended for README-level status reporting:
- It runs a short fixed-boundary solve on bundled inputs using vmec_jax defaults.
- It compares a few "end-to-end" outputs against bundled VMEC reference wouts.

This complements the solver-free pipeline snapshot in `pipeline_parity_summary.py`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import vmec_jax.api as vj


def _rel_rms(x: np.ndarray, y: np.ndarray, *, eps: float = 1e-16) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    num = float(np.sqrt(np.mean((x - y) ** 2)))
    denom = float(np.sqrt(np.mean(y**2)))
    return num / max(eps, denom)


def _format(x: float) -> str:
    if not np.isfinite(x):
        return "nan"
    return f"{x:.2e}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--cases",
        nargs="*",
        default=["circular_tokamak", "shaped_tokamak_pressure", "solovev"],
    )
    p.add_argument("--solver", default="vmec2000_iter")
    p.add_argument("--max-iter", type=int, default=30)
    p.add_argument(
        "--use-input-niter",
        action="store_true",
        help="For vmec2000_iter: respect NITER_ARRAY/FTOL_ARRAY staging (still capped by --max-iter).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable vmec_jax solver iteration prints.",
    )
    args = p.parse_args()

    root = Path(__file__).resolve().parents[2]
    data_dir = root / "examples" / "data"

    cases = [str(c) for c in args.cases]

    print("| Case | input | ns | mpol | ntor | nfp | solver | max_iter | ftol | fsq_total(ref) | fsq_total(new) | rmnc relRMS | zmns relRMS |")
    print("|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|")

    for case in cases:
        input_path = data_dir / f"input.{case}"
        wout_path = data_dir / f"wout_{case}_reference.nc"
        if not wout_path.exists():
            wout_path = data_dir / f"wout_{case}.nc"
        if not input_path.exists() or not wout_path.exists():
            raise FileNotFoundError(f"Missing bundled input/wout for case={case!r}")

        cfg, indata = vj.load_input(input_path)
        wref = vj.read_wout(wout_path)
        ftol = float(indata.get_float("FTOL", 1e-10))

        run = vj.run_fixed_boundary(
            input_path,
            solver=str(args.solver),
            max_iter=int(args.max_iter),
            multigrid_use_input_niter=bool(args.use_input_niter),
            verbose=bool(args.verbose),
        )
        fsqr, fsqz, fsql = vj.residual_scalars_from_state(
            state=run.state,
            static=run.static,
            indata=run.indata,
            signgs=int(run.signgs),
            use_vmec_synthesis=True,
        )
        fsq_new = float(fsqr + fsqz + fsql)
        fsq_ref = float(wref.fsqr + wref.fsqz + wref.fsql)

        rmnc_err = _rel_rms(np.asarray(run.state.Rcos), np.asarray(wref.rmnc))
        zmns_err = _rel_rms(np.asarray(run.state.Zsin), np.asarray(wref.zmns))

        print(
            "| "
            + " | ".join(
                [
                    case,
                    f"`{input_path.name}`",
                    str(int(cfg.ns)),
                    str(int(cfg.mpol)),
                    str(int(cfg.ntor)),
                    str(int(cfg.nfp)),
                    str(args.solver),
                    str(int(args.max_iter)),
                    _format(ftol),
                    _format(fsq_ref),
                    _format(fsq_new),
                    _format(rmnc_err),
                    _format(zmns_err),
                ]
            )
            + " |"
        )

    print()
    print("Notes:")
    print("- This is an end-to-end *solver* snapshot. It is not expected to match VMEC2000 yet on all cases.")
    print("- `fsq_total(new)` is computed via vmec_jax scalar residual kernels on the final iterate.")
    print("- `rmnc/zmns relRMS` compare Fourier coefficients directly (main modes).")
    print("- `ftol` is read from the input namelist (`FTOL`).")


if __name__ == "__main__":
    main()
