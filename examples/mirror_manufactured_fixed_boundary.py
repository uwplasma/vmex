"""Run a manufactured fixed-boundary mirror solve with a known stationary state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vmec_jax.mirror import MirrorResolution, MirrorStateAxisym
from vmec_jax.mirror.kernels.fields import evaluate_axisym_field
from vmec_jax.mirror.kernels.geometry import evaluate_axisym_geometry
from vmec_jax.mirror.validation.manufactured import make_mms_case, solve_axisym_mms_fixed_boundary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("results/mirror/manufactured_fixed_boundary"))
    parser.add_argument("--ns", type=int, default=5)
    parser.add_argument("--nxi", type=int, default=9)
    parser.add_argument("--maxiter", type=int, default=20)
    parser.add_argument("--gtol", type=float, default=1.0e-12)
    parser.add_argument("--ftol", type=float, default=1.0e-12)
    parser.add_argument("--perturbation", type=float, default=0.002)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def _initial_state(case, *, amplitude: float) -> MirrorStateAxisym:
    grid = case.grid
    s = grid.s_full[:, None]
    xi = grid.xi[None, :]
    shape = s * (1.0 - s) * (1.0 - xi**2)
    a = case.state.a * (1.0 + float(amplitude) * shape)
    lam = case.state.lam + 0.1 * float(amplitude) * shape * xi
    return MirrorStateAxisym(a=a, lam=lam)


def _write_residual_plot(result, *, outdir: Path) -> Path:
    import matplotlib.pyplot as plt

    iterations = np.asarray([row.iteration for row in result.trace], dtype=int)
    residual = np.asarray([row.residual_norm for row in result.trace], dtype=float)
    fsq = np.asarray([row.fsq for row in result.trace], dtype=float)
    error = np.asarray([row.exact_error_norm for row in result.trace], dtype=float)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.semilogy(iterations, residual, "o-", label="reduced residual")
    ax.semilogy(iterations, fsq, "s-", label="manufactured fsq")
    ax.semilogy(iterations, error, "^-", label="exact-state error")
    ax.set_xlabel("iteration")
    ax.set_ylabel("norm")
    ax.set_title("manufactured fixed-boundary convergence")
    ax.legend(fontsize="small")
    fig.tight_layout()
    path = outdir / "manufactured_fixed_boundary_residual.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _write_geometry_overlay(result, *, outdir: Path) -> Path:
    import matplotlib.pyplot as plt

    grid = result.case.grid
    rho = grid.rho_full[:, None]
    exact_r = rho * result.case.state.a
    solved_r = rho * result.state.a
    z = grid.z

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    for idx in np.linspace(1, grid.ns - 1, min(5, grid.ns - 1), dtype=int):
        ax.plot(z, exact_r[idx], "k-", linewidth=1.1, alpha=0.65)
        ax.plot(z, solved_r[idx], "r--", linewidth=1.0, alpha=0.8)
        ax.plot(z, -exact_r[idx], "k-", linewidth=1.1, alpha=0.65)
        ax.plot(z, -solved_r[idx], "r--", linewidth=1.0, alpha=0.8)
    ax.set_xlabel("z")
    ax.set_ylabel("x")
    ax.set_title("exact and solved flux surfaces")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    path = outdir / "manufactured_fixed_boundary_geometry.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _write_bmag_map(result, *, outdir: Path) -> Path:
    import matplotlib.pyplot as plt

    grid = result.case.grid
    geometry = evaluate_axisym_geometry(result.state, grid)
    field = evaluate_axisym_field(
        result.state,
        grid,
        geometry,
        psi_prime=result.case.psi_prime,
        i_prime=result.case.i_prime,
    )
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    mesh = ax.pcolormesh(grid.z, grid.s_full, field.bmag, shading="auto")
    ax.set_xlabel("z")
    ax.set_ylabel("s")
    ax.set_title("manufactured solved |B|")
    fig.colorbar(mesh, ax=ax, label="|B|")
    fig.tight_layout()
    path = outdir / "manufactured_fixed_boundary_bmag_sxi.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _write_boundary_3d_plot(result, *, outdir: Path) -> Path:
    import matplotlib.pyplot as plt

    grid = result.case.grid
    geometry = evaluate_axisym_geometry(result.state, grid)
    field = evaluate_axisym_field(
        result.state,
        grid,
        geometry,
        psi_prime=result.case.psi_prime,
        i_prime=result.case.i_prime,
    )
    theta = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=True)
    z = grid.z
    radius = geometry.r[-1]
    zz, tt = np.meshgrid(z, theta)
    rr = np.broadcast_to(radius[None, :], zz.shape)
    bmag = np.broadcast_to(field.bmag[-1][None, :], zz.shape)
    x = rr * np.cos(tt)
    y = rr * np.sin(tt)

    fig = plt.figure(figsize=(7.0, 4.8))
    ax = fig.add_subplot(111, projection="3d")
    colors = plt.cm.viridis((bmag - np.min(bmag)) / max(np.ptp(bmag), np.finfo(float).tiny))
    surf = ax.plot_surface(zz, x, y, facecolors=colors, linewidth=0.0, antialiased=True, shade=False)
    surf.set_array(bmag.ravel())
    surf.set_cmap("viridis")
    ax.set_xlabel("z")
    ax.set_ylabel("x")
    ax.set_zlabel("y")
    ax.set_title("manufactured boundary |B|")
    ax.view_init(elev=18, azim=-62)
    fig.colorbar(surf, ax=ax, shrink=0.72, label="|B|")
    fig.tight_layout()
    path = outdir / "manufactured_fixed_boundary_boundary_3d.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def run_case(
    outdir: Path,
    *,
    ns: int = 5,
    nxi: int = 9,
    maxiter: int = 20,
    gtol: float = 1.0e-12,
    ftol: float = 1.0e-12,
    perturbation: float = 0.002,
    write_plots: bool = True,
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    case = make_mms_case("axisym_projected_fixed_boundary", MirrorResolution(ns=ns, ntheta=1, nxi=nxi, mpol=0), mu0=1.0)
    initial_state = _initial_state(case, amplitude=perturbation)
    result = solve_axisym_mms_fixed_boundary(
        case,
        initial_state=initial_state,
        maxiter=maxiter,
        gtol=gtol,
        ftol=ftol,
        line_search_steps=32,
        mu0=1.0,
    )
    if write_plots:
        _write_residual_plot(result, outdir=outdir)
        _write_geometry_overlay(result, outdir=outdir)
        _write_bmag_map(result, outdir=outdir)
        _write_boundary_3d_plot(result, outdir=outdir)

    metrics = {
        "ns": int(ns),
        "nxi": int(nxi),
        "maxiter": int(maxiter),
        "gtol": float(gtol),
        "ftol": float(ftol),
        "optimizer_success": bool(result.optimizer_success),
        "optimizer_status": int(result.optimizer_status),
        "optimizer_message": result.optimizer_message,
        "optimizer_nit": int(result.optimizer_nit),
        "optimizer_nfev": int(result.optimizer_nfev),
        "optimizer_njev": int(result.optimizer_njev),
        "initial_residual_norm": float(result.trace[0].residual_norm),
        "final_residual_norm": float(result.residual_norm),
        "residual_drop": float(result.trace[0].residual_norm - result.residual_norm),
        "final_fsq": float(result.fsq),
        "initial_exact_error_norm": float(result.trace[0].exact_error_norm),
        "final_exact_error_norm": float(result.exact_error_norm),
        "reached_projected_gtol": bool(result.residual_norm <= float(gtol)),
    }
    path = outdir / "manufactured_fixed_boundary_metrics.json"
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = run_case(
        args.outdir,
        ns=args.ns,
        nxi=args.nxi,
        maxiter=args.maxiter,
        gtol=args.gtol,
        ftol=args.ftol,
        perturbation=args.perturbation,
        write_plots=not args.no_plots,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
