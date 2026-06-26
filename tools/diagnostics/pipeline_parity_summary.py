"""Fixed-boundary pipeline parity snapshot (fast; no solve).

This script evaluates the *same intermediate quantities* that appear in VMEC's
fixed-boundary kernels on a small set of bundled reference `wout_*.nc` files,
then compares against the corresponding quantities stored in the reference wout.

It is designed to generate a README-friendly, 4-case snapshot table without
running the nonlinear fixed-boundary iteration.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import numpy as np

import vmec_jax.api as vj
from vmec_jax.config import load_config
from vmec_jax.fourier import build_helical_basis, eval_fourier
from vmec_jax.grids import AngleGrid
from vmec_jax.modes import ModeTable
from vmec_jax.static import build_static
from vmec_jax.kernels.bcovar import vmec_bcovar_half_mesh_from_wout
from vmec_jax.kernels.tomnsp import vmec_angle_grid
from vmec_jax.wout import read_wout, state_from_wout


def _rel_rms(x: np.ndarray, y: np.ndarray, *, eps: float = 1e-16) -> float:
    x = np.asarray(x)
    y = np.asarray(y)
    num = float(np.sqrt(np.mean((x - y) ** 2)))
    denom = float(np.sqrt(np.mean(y**2)))
    return num / max(eps, denom)


def _rel_err(x: float, y: float, *, eps: float = 1e-16) -> float:
    return abs(float(x) - float(y)) / max(eps, abs(float(y)))


def _format(x: float) -> str:
    if not np.isfinite(x):
        return "nan"
    return f"{x:.2e}"


def _case_parity(*, input_path: Path, wout_path: Path, jit: bool) -> dict[str, float]:
    cfg, indata = load_config(str(input_path))
    wout = read_wout(wout_path)
    state = state_from_wout(wout)

    # Some bundled reference wouts were generated with higher-order Fourier
    # settings than the low-res input files. Ensure the static/mode tables match
    # the wout we are validating against.
    if (
        int(cfg.mpol) != int(wout.mpol)
        or int(cfg.ntor) != int(wout.ntor)
        or int(cfg.nfp) != int(wout.nfp)
        or bool(cfg.lasym) != bool(wout.lasym)
        or int(cfg.ns) != int(wout.ns)
    ):
        cfg = replace(
            cfg,
            mpol=int(wout.mpol),
            ntor=int(wout.ntor),
            nfp=int(wout.nfp),
            lasym=bool(wout.lasym),
            ns=int(wout.ns),
        )

    grid = vmec_angle_grid(
        ntheta=int(cfg.ntheta),
        nzeta=int(cfg.nzeta),
        nfp=int(cfg.nfp),
        lasym=bool(cfg.lasym),
    )
    static = build_static(cfg, grid=grid)

    try:
        import jax
    except Exception:
        jax = None

    ctx = (jax.disable_jit() if (jax is not None and not jit) else nullcontext())
    with ctx:
        # Reference Nyquist fields from the wout, evaluated on the active angle grid.
        grid_eval = AngleGrid(theta=np.asarray(static.grid.theta), zeta=np.asarray(static.grid.zeta), nfp=int(wout.nfp))
        modes_nyq = ModeTable(m=wout.xm_nyq, n=(wout.xn_nyq // wout.nfp))
        basis_nyq = build_helical_basis(modes_nyq, grid_eval)

        sqrtg_ref = np.asarray(eval_fourier(wout.gmnc, wout.gmns, basis_nyq))
        bsupu_ref = np.asarray(eval_fourier(wout.bsupumnc, wout.bsupumns, basis_nyq))
        bsupv_ref = np.asarray(eval_fourier(wout.bsupvmnc, wout.bsupvmns, basis_nyq))
        bsubu_ref = np.asarray(eval_fourier(wout.bsubumnc, wout.bsubumns, basis_nyq))
        bsubv_ref = np.asarray(eval_fourier(wout.bsubvmnc, wout.bsubvmns, basis_nyq))
        bmag_ref = np.asarray(eval_fourier(wout.bmnc, wout.bmns, basis_nyq))
        bsq_ref = 0.5 * (bmag_ref**2) + np.asarray(wout.pres)[:, None, None]

        # VMEC-like half-mesh pipeline reconstructed from the reference state.
        bc = vmec_bcovar_half_mesh_from_wout(
            state=state,
            static=static,
            wout=wout,
            use_vmec_synthesis=True,
            use_wout_bsup=False,
            use_wout_bsub_for_lambda=False,
            use_wout_bmag_for_bsq=False,
        )
        sqrtg_new = np.asarray(bc.jac.sqrtg)
        bsupu_new = np.asarray(bc.bsupu)
        bsupv_new = np.asarray(bc.bsupv)
        bsubu_new = np.asarray(bc.bsubu)
        bsubv_new = np.asarray(bc.bsubv)

        # VMEC wout stores bsupu with the opposite sign of the internal bcovar
        # convention (signgs orientation). For solver-free parity, align the
        # sign so the comparison is meaningful.
        if np.sign(np.sum(bsupu_new * bsupu_ref)) < 0.0:
            bsupu_new = -bsupu_new
            guu = np.asarray(bc.guu)
            guv = np.asarray(bc.guv)
            gvv = np.asarray(bc.gvv)
            bsubu_new = guu * bsupu_new + guv * bsupv_new
            bsubv_new = guv * bsupu_new + gvv * bsupv_new

        b2_new = bsupu_new * bsubu_new + bsupv_new * bsubv_new
        bmag_new = np.sqrt(np.maximum(0.0, b2_new))
        bsq_new = np.asarray(bc.bsq)

        fsqr_new, fsqz_new, fsql_new = vj.residual_scalars_from_state(
            state=state,
            static=static,
            indata=indata,
            signgs=int(wout.signgs),
            wout=wout,
            use_vmec_synthesis=True,
        )

    out: dict[str, float] = {}
    out["sqrtg"] = _rel_rms(sqrtg_new, sqrtg_ref)
    out["bsupu"] = _rel_rms(bsupu_new, bsupu_ref)
    out["bsupv"] = _rel_rms(bsupv_new, bsupv_ref)
    out["bsubu"] = _rel_rms(bsubu_new, bsubu_ref)
    out["bsubv"] = _rel_rms(bsubv_new, bsubv_ref)
    out["absB"] = _rel_rms(bmag_new, bmag_ref)
    out["bsq"] = _rel_rms(bsq_new, bsq_ref)

    out["fsqr"] = _rel_err(float(fsqr_new), float(wout.fsqr))
    out["fsqz"] = _rel_err(float(fsqz_new), float(wout.fsqz))
    out["fsql"] = _rel_err(float(fsql_new), float(wout.fsql))
    fsq_total_new = float(fsqr_new + fsqz_new + fsql_new)
    fsq_total_ref = float(wout.fsqr + wout.fsqz + wout.fsql)
    out["fsq_total"] = _rel_err(fsq_total_new, fsq_total_ref)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--cases",
        nargs="*",
        default=["circular_tokamak", "purely_toroidal_field", "shaped_tokamak_pressure", "solovev"],
        help="Case names from examples/data as input.<case> and wout_<case>_reference.nc.",
    )
    p.add_argument(
        "--jit",
        action="store_true",
        help="Enable JAX JIT (may add multi-minute compile on first run).",
    )
    p.add_argument(
        "--markdown",
        action="store_true",
        help="Print a GitHub-flavored Markdown table (default).",
    )
    args = p.parse_args()

    root = Path(__file__).resolve().parents[2]
    data_dir = root / "examples" / "data"

    cases = [str(c) for c in args.cases]
    metrics = ["sqrtg", "bsupu", "bsupv", "bsubu", "bsubv", "absB", "bsq", "fsqr", "fsqz", "fsql", "fsq_total"]
    rows: dict[str, list[float]] = {m: [] for m in metrics}

    for case in cases:
        input_path = data_dir / f"input.{case}"
        wout_path = data_dir / f"wout_{case}_reference.nc"
        if not wout_path.exists():
            wout_path = data_dir / f"wout_{case}.nc"
        if not input_path.exists() or not wout_path.exists():
            raise FileNotFoundError(f"Missing bundled input/wout for case={case!r}")

        d = _case_parity(input_path=input_path, wout_path=wout_path, jit=bool(args.jit))
        for m in metrics:
            rows[m].append(float(d[m]))

    # Markdown table output by default (README-friendly).
    print("| Variable | " + " | ".join(cases) + " |")
    print("|---| " + " | ".join([":--:" for _ in cases]) + " |")
    names = {
        "sqrtg": "sqrtg",
        "bsupu": "bsupu",
        "bsupv": "bsupv",
        "bsubu": "bsubu",
        "bsubv": "bsubv",
        "absB": "abs(B)",
        "bsq": "bsq = 0.5*B^2 + p",
        "fsqr": "fsqr",
        "fsqz": "fsqz",
        "fsql": "fsql",
        "fsq_total": "fsq_total",
    }
    for m in metrics:
        vals = " | ".join(_format(v) for v in rows[m])
        print(f"| {names[m]} | {vals} |")

    print()
    print("Notes:")
    print("- Array rows are relative RMS on the VMEC angle grid.")
    print("- Scalar rows compare vmec_jax kernels against `wout.fsqr/fsqz/fsql`.")
    print("- This script does not run a nonlinear fixed-boundary solve.")


if __name__ == "__main__":
    main()
