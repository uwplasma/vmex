"""Run a low-resolution fixed-boundary mirror with a theta-shaped side wall."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vmec_jax.mirror import (
    IPrimeProfile,
    MirrorBoundary,
    MirrorConfig,
    MirrorResolution,
    MirrorSolveOptions,
    PressureProfile,
    PsiPrimeProfile,
    plot_mirror_output,
    run_mirror_fixed_boundary,
    write_mirror_output,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("results/mirror/nonaxisymmetric_boundary"))
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--maxiter", type=int, default=6)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def run_case(outdir: Path, *, epsilon: float, maxiter: int, write_plots: bool = True) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    config = MirrorConfig(MirrorResolution(ns=7, ntheta=13, nxi=17, mpol=4), z_min=-1.6, z_max=1.6)
    boundary = MirrorBoundary.cosine_modulated_radius(r0=0.3, a2=-0.35, epsilon=epsilon, theta_mode=2)
    result = run_mirror_fixed_boundary(
        config,
        boundary,
        psi_prime=PsiPrimeProfile.constant(0.01),
        i_prime=IPrimeProfile.zero(),
        pressure=PressureProfile.zero(),
        options=MirrorSolveOptions(optimizer="lbfgs", maxiter=maxiter, tolerance=1.0e-10, mu0=1.0),
    )
    mout = write_mirror_output(outdir / "mout_nonaxisymmetric_boundary.nc", result, overwrite=True)
    if write_plots:
        plot_mirror_output(mout, outdir=outdir / "figures")
    return mout


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    mout = run_case(args.outdir, epsilon=args.epsilon, maxiter=args.maxiter, write_plots=not args.no_plots)
    print(mout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
