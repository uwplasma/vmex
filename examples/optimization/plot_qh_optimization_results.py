#!/usr/bin/env python

"""Plot results from the QH fixed-resolution exact-adjoint optimization.

Reads the wout files and history JSON produced by
``qh_fixed_resolution_exact.py --output-dir <dir>`` and generates:

  - ``boundary_comparison.png``   3D LCFS coloured by |B| (initial vs final)
  - ``bmag_surface.png``          |B| on LCFS unrolled to (theta, phi)
  - ``objective_history.png``     Objective value and aspect ratio vs iteration

Usage
-----
Run the optimization first::

    python examples/optimization/qh_fixed_resolution_exact.py \\
        --output-dir results/qh_opt --max-mode 1 --max-nfev 10

Then plot::

    python examples/optimization/plot_qh_optimization_results.py \\
        --output-dir results/qh_opt

All figures are saved inside ``--output-dir``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import vmec_jax as vj
from vmec_jax.plotting import vmecplot2_lcfs_3d_grid, vmecplot2_bmag_grid


def _parse_args():
    p = argparse.ArgumentParser(description="Plot QH optimization results")
    p.add_argument(
        "--output-dir",
        type=str,
        default="results/qh_opt",
        help="Directory containing wout_initial.nc, wout_final.nc, history.json",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        default=True,
        help="Do not call plt.show() (default: True for non-interactive use)",
    )
    return p.parse_args()


def _lcfs_xyz(R: np.ndarray, Z: np.ndarray, phi: np.ndarray):
    """Convert cylindrical (R, Z, phi) LCFS grids to Cartesian (X, Y, Z)."""
    # R, Z shape: (ntheta, nphi); phi shape: (nphi,)
    X = R * np.cos(phi[None, :])
    Y = R * np.sin(phi[None, :])
    return X, Y, Z


def plot_3d_boundary_comparison(
    wout_init,
    wout_final,
    outdir: Path,
) -> Path:
    """3-D LCFS plots coloured by |B|, initial (left) vs final (right)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    ns_init = int(np.asarray(wout_init.ns))
    ns_final = int(np.asarray(wout_final.ns))
    s_idx_init = ns_init - 1
    s_idx_final = ns_final - 1

    theta_i, phi_i, R_i, Z_i, B_i = vmecplot2_lcfs_3d_grid(wout_init, s_index=s_idx_init, ntheta=60, nzeta=None)
    theta_f, phi_f, R_f, Z_f, B_f = vmecplot2_lcfs_3d_grid(wout_final, s_index=s_idx_final, ntheta=60, nzeta=None)

    X_i, Y_i, _ = _lcfs_xyz(R_i, Z_i, phi_i)
    X_f, Y_f, _ = _lcfs_xyz(R_f, Z_f, phi_f)

    # Shared colour scale across both panels.
    Bmin = min(B_i.min(), B_f.min())
    Bmax = max(B_i.max(), B_f.max())
    norm = Normalize(vmin=Bmin, vmax=Bmax)
    cmap = plt.cm.viridis

    fig = plt.figure(figsize=(12, 5))

    for col, (X, Y, Z_plt, B, title) in enumerate([
        (X_i, Y_i, Z_i, B_i, "Initial boundary"),
        (X_f, Y_f, Z_f, B_f, "Optimized boundary"),
    ]):
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")
        fcolors = cmap(norm(B))
        ax.plot_surface(X, Y, Z_plt, facecolors=fcolors, rstride=1, cstride=1,
                        linewidth=0, antialiased=False, shade=False)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(title, fontsize=11)
        ax.set_box_aspect([1, 1, 0.5])

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=fig.axes, label="|B| (T)", shrink=0.6, pad=0.1)
    fig.suptitle("QH LCFS coloured by |B| — nfp=4", fontsize=13, y=1.01)

    out = outdir / "boundary_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")
    return out


def plot_bmag_surface(
    wout_init,
    wout_final,
    outdir: Path,
) -> Path:
    """Unrolled |B|(theta, phi) on LCFS — initial (top) vs final (bottom)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns_init = int(np.asarray(wout_init.ns))
    ns_final = int(np.asarray(wout_final.ns))
    nfp = int(np.asarray(wout_init.nfp))

    theta_i, zeta_i, B_i = vmecplot2_bmag_grid(wout_init, s_index=ns_init - 1, ntheta=64, nzeta=128)
    theta_f, zeta_f, B_f = vmecplot2_bmag_grid(wout_final, s_index=ns_final - 1, ntheta=64, nzeta=128)

    # Shared colour limits.
    vmin = min(B_i.min(), B_f.min())
    vmax = max(B_i.max(), B_f.max())

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for ax, B, theta, zeta, title in [
        (axes[0], B_i, theta_i, zeta_i, "Initial"),
        (axes[1], B_f, theta_f, zeta_f, "Optimized"),
    ]:
        im = ax.pcolormesh(
            np.degrees(zeta) / nfp,
            np.degrees(theta),
            B,
            shading="auto",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_ylabel("Poloidal angle θ (°)")
        ax.set_title(f"|B| on LCFS — {title}", fontsize=11)
        fig.colorbar(im, ax=ax, label="|B| (T)")
    axes[-1].set_xlabel("Toroidal angle φ/nfp (°)")
    fig.suptitle("Magnetic field strength on LCFS", fontsize=13)
    fig.tight_layout()

    out = outdir / "bmag_surface.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")
    return out


def plot_objective_history(history_path: Path, outdir: Path) -> Path:
    """Objective value and aspect ratio vs. Jacobian evaluation number."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(history_path) as f:
        data = json.load(f)

    hist = data["history"]
    objectives = [h["objective"] for h in hist]
    aspects = [h["aspect"] for h in hist]
    wall_times = [h["wall_time_s"] for h in hist]
    iters = list(range(len(hist)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    ax1.semilogy(iters, objectives, "o-", color="steelblue", linewidth=2, markersize=6)
    ax1.set_ylabel("Objective  Σ residuals²", fontsize=11)
    ax1.set_title(
        f"QH optimization  (max_mode={data['max_mode']}, {data['nfev']} evals, "
        f"{data['total_wall_time_s']:.0f} s)",
        fontsize=11,
    )
    ax1.axhline(objectives[-1], color="steelblue", linestyle="--", alpha=0.4,
                label=f"Final: {objectives[-1]:.4f}")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(iters, aspects, "s-", color="darkorange", linewidth=2, markersize=6)
    ax2.axhline(7.0, color="k", linestyle=":", alpha=0.5, label="Target A=7")
    ax2.set_ylabel("Aspect ratio", fontsize=11)
    ax2.set_xlabel("Jacobian evaluation index", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out = outdir / "objective_history.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")
    return out


def main() -> None:
    args = _parse_args()
    outdir = Path(args.output_dir)

    wout_init_path = outdir / "wout_initial.nc"
    wout_final_path = outdir / "wout_final.nc"
    history_path = outdir / "history.json"

    for p in (wout_init_path, wout_final_path, history_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found.  Run qh_fixed_resolution_exact.py first:\n"
                f"  python examples/optimization/qh_fixed_resolution_exact.py "
                f"--output-dir {outdir}"
            )

    print(f"Loading wout files from {outdir} …")
    wout_init = vj.load_wout(str(wout_init_path))
    wout_final = vj.load_wout(str(wout_final_path))

    print("Generating plots …")
    plot_3d_boundary_comparison(wout_init, wout_final, outdir)
    plot_bmag_surface(wout_init, wout_final, outdir)
    plot_objective_history(history_path, outdir)

    with open(history_path) as f:
        data = json.load(f)
    print()
    print(f"Summary: {data['nfev']} residual evals in {data['total_wall_time_s']:.1f} s")
    print(f"  Objective: {data['objective_initial']:.4f}  →  {data['objective_final']:.4f}")
    print(f"  QS total:  {data['qs_initial']:.4f}  →  {data['qs_final']:.4f}")
    print(f"  Aspect:    {data['aspect_initial']:.3f}  →  {data['aspect_final']:.3f}")
    print("Done.")


if __name__ == "__main__":
    main()
