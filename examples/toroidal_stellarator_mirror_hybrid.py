"""Build a toroidal stellarator-mirror hybrid fixed-boundary input."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np

import vmec_jax as vj
from vmec_jax.namelist import write_indata
from vmec_jax.toroidal_hybrid import (
    sample_toroidal_stellarator_mirror_hybrid_boundary,
    toroidal_hybrid_cross_section_anisotropy,
    toroidal_hybrid_cross_section_orientation,
    toroidal_stellarator_mirror_hybrid_indata,
    toroidal_stellarator_mirror_hybrid_metrics,
)


def _import_matplotlib():
    try:
        mpl_cache = Path(tempfile.gettempdir()) / "vmec_jax_mplconfig"
        mpl_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
        vj.fix_matplotlib_3d  # trigger lazy plotting import availability checks
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
    except Exception as exc:  # pragma: no cover
        raise SystemExit("This example requires matplotlib.") from exc
    return plt, Normalize, ScalarMappable


def _full_torus_xyz(samples, *, nfp: int):
    phi_parts = [(samples.zeta + 2.0 * np.pi * period) / float(nfp) for period in range(int(nfp))]
    phi = np.concatenate(phi_parts)
    R = np.tile(samples.R, (1, int(nfp)))
    Z = np.tile(samples.Z, (1, int(nfp)))
    corner = np.tile(samples.corner_weight, (1, int(nfp)))
    X = R * np.cos(phi[None, :])
    Y = R * np.sin(phi[None, :])
    return X, Y, Z, corner


def _write_boundary_plots(samples, *, outdir: Path, nfp: int) -> dict[str, str]:
    plt, Normalize, ScalarMappable = _import_matplotlib()
    outdir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    X, Y, Z, corner = _full_torus_xyz(samples, nfp=nfp)
    norm = Normalize(vmin=0.0, vmax=1.0)
    colors = plt.cm.viridis(norm(corner))
    fig = plt.figure(figsize=(7.2, 5.8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, facecolors=colors, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
    ax.set_title("Toroidal hybrid LCFS")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    vj.fix_matplotlib_3d(ax)
    fig.colorbar(ScalarMappable(norm=norm, cmap="viridis"), ax=ax, shrink=0.72, pad=0.08, label="corner weight")
    path = outdir / "toroidal_hybrid_lcfs_3d.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths["lcfs_3d"] = str(path)

    fig, ax = plt.subplots(1, 1, figsize=(6.2, 6.2), constrained_layout=True)
    for theta_idx, label in [(0, "outer"), (samples.theta.size // 2, "inner")]:
        x_line = X[theta_idx, :]
        y_line = Y[theta_idx, :]
        ax.plot(x_line, y_line, lw=1.4, label=label)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Top view")
    ax.legend(loc="best")
    path = outdir / "toroidal_hybrid_top_view.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths["top_view"] = str(path)

    side_cols = [0, samples.zeta.size // 2]
    corner_cols = [samples.zeta.size // 4, (3 * samples.zeta.size) // 4]
    fig, ax = plt.subplots(1, 1, figsize=(6.2, 5.2), constrained_layout=True)
    for col in side_cols:
        r_closed, z_closed = _closed_curve(samples.R[:, col], samples.Z[:, col])
        ax.plot(r_closed, z_closed, color="tab:blue", lw=1.5, alpha=0.8)
    for col in corner_cols:
        r_closed, z_closed = _closed_curve(samples.R[:, col], samples.Z[:, col])
        ax.plot(r_closed, z_closed, color="tab:red", lw=1.5, alpha=0.8)
    ax.plot([], [], color="tab:blue", label="mirror-like side arcs")
    ax.plot([], [], color="tab:red", label="stellarator-like corner arcs")
    ax.set_aspect("equal", "box")
    ax.set_xlabel("R")
    ax.set_ylabel("Z")
    ax.set_title("Side and corner cross sections")
    ax.legend(loc="best")
    path = outdir / "toroidal_hybrid_cross_sections.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths["cross_sections"] = str(path)

    orientation = toroidal_hybrid_cross_section_orientation(samples)
    anisotropy = toroidal_hybrid_cross_section_anisotropy(samples)
    anisotropy_threshold = 1.0e-14 + 1.0e-8 * float(np.max(anisotropy))
    valid_orientation = anisotropy > anisotropy_threshold
    side_weight = np.mean(samples.side_weight, axis=0)
    corner_weight = np.mean(samples.corner_weight, axis=0)
    fig, ax0 = plt.subplots(1, 1, figsize=(6.8, 4.2), constrained_layout=True)
    ax0.plot(
        samples.zeta[valid_orientation],
        orientation[valid_orientation],
        ".",
        color="tab:purple",
        ms=4,
        label="valid ellipse orientation",
    )
    if np.any(~valid_orientation):
        ax0.plot(
            samples.zeta[~valid_orientation],
            orientation[~valid_orientation],
            "x",
            color="0.45",
            ms=5,
            label="undefined covariance axis",
        )
    ax0.set_xlabel("zeta")
    ax0.set_ylabel("principal-axis angle")
    ax1 = ax0.twinx()
    ax1.plot(samples.zeta, side_weight, "--", color="tab:blue", lw=1.4, label="side weight")
    ax1.plot(samples.zeta, corner_weight, "--", color="tab:red", lw=1.4, label="corner weight")
    ax1.set_ylabel("region weight")
    lines0, labels0 = ax0.get_legend_handles_labels()
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax0.legend(lines0 + lines1, labels0 + labels1, loc="best", fontsize="small")
    ax0.set_title("Mirror-side and stellarator-corner regions")
    path = outdir / "toroidal_hybrid_region_orientation.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths["region_orientation"] = str(path)
    return paths


def _closed_curve(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.r_[x, x[:1]], np.r_[y, y[:1]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=str, default="results/toroidal_stellarator_mirror_hybrid")
    parser.add_argument("--nfp", type=int, default=2)
    parser.add_argument("--mpol", type=int, default=5)
    parser.add_argument("--ntor", type=int, default=4)
    parser.add_argument("--ns", type=int, default=15)
    parser.add_argument("--niter", type=int, default=80)
    parser.add_argument("--ftol", type=float, default=1.0e-9)
    parser.add_argument("--major-radius", type=float, default=1.15)
    parser.add_argument("--minor-radius", type=float, default=0.18)
    parser.add_argument("--axis-oval", type=float, default=0.10)
    parser.add_argument("--side-minor-modulation", type=float, default=0.10)
    parser.add_argument("--side-elongation", type=float, default=0.28)
    parser.add_argument("--corner-amplitude", type=float, default=0.035)
    parser.add_argument("--corner-helicity", type=int, default=1)
    parser.add_argument("--side-power", type=float, default=1.0)
    parser.add_argument("--corner-power", type=float, default=1.0)
    parser.add_argument("--ntheta-fit", type=int, default=64)
    parser.add_argument("--nzeta-fit", type=int, default=64)
    parser.add_argument("--run-solve", action="store_true", help="Run vmec_jax after writing the input file.")
    parser.add_argument("--max-iter", type=int, default=3, help="Iteration cap used only with --run-solve.")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    sample_kwargs = {
        "major_radius": float(args.major_radius),
        "minor_radius": float(args.minor_radius),
        "axis_oval": float(args.axis_oval),
        "side_minor_modulation": float(args.side_minor_modulation),
        "side_elongation": float(args.side_elongation),
        "corner_amplitude": float(args.corner_amplitude),
        "corner_helicity": int(args.corner_helicity),
        "side_power": float(args.side_power),
        "corner_power": float(args.corner_power),
    }
    samples = sample_toroidal_stellarator_mirror_hybrid_boundary(
        ntheta=int(args.ntheta_fit),
        nzeta=int(args.nzeta_fit),
        **sample_kwargs,
    )
    indata = toroidal_stellarator_mirror_hybrid_indata(
        nfp=int(args.nfp),
        mpol=int(args.mpol),
        ntor=int(args.ntor),
        ntheta_fit=int(args.ntheta_fit),
        nzeta_fit=int(args.nzeta_fit),
        ns_array=int(args.ns),
        niter_array=int(args.niter),
        ftol_array=float(args.ftol),
        **sample_kwargs,
    )
    input_path = outdir / "input.toroidal_stellarator_mirror_hybrid"
    write_indata(input_path, indata)

    metrics = toroidal_stellarator_mirror_hybrid_metrics(samples)
    metrics.update(
        {
            "input": str(input_path),
            "nfp": int(args.nfp),
            "mpol": int(args.mpol),
            "ntor": int(args.ntor),
            "rbc_count": len(indata.indexed.get("RBC", {})),
            "zbs_count": len(indata.indexed.get("ZBS", {})),
            "sample_parameters": sample_kwargs,
            "figures": {},
        }
    )
    if not bool(args.no_plots):
        metrics["figures"] = _write_boundary_plots(samples, outdir=outdir / "figures", nfp=int(args.nfp))

    if bool(args.run_solve):
        run = vj.run_fixed_boundary(
            input_path,
            solver="vmec2000_iter",
            solver_mode="accelerated",
            max_iter=int(args.max_iter),
            cli_fixed_boundary_mode=True,
            verbose=False,
        )
        wout_path = outdir / "wout_toroidal_stellarator_mirror_hybrid.nc"
        vj.write_wout_from_fixed_boundary_run(wout_path, run, include_fsq=True)
        metrics["wout"] = str(wout_path)
        if not bool(args.no_plots):
            metrics["wout_figures"] = {
                key: str(value) for key, value in vj.plot_wout(wout_path, outdir=outdir / "figures" / "wout").items()
            }

    metrics_path = outdir / "toroidal_stellarator_mirror_hybrid_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(metrics_path)


if __name__ == "__main__":  # pragma: no cover
    main()
