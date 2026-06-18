"""Sample circular-coil fields for the mirror free-boundary planning lane."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vmec_jax.mirror import (
    MirrorCircularCoils,
    initial_mirror_boundary_from_circular_coil_scan,
    make_mirror_free_boundary_circular_coil_scan,
    make_mirror_grid,
    sample_mirror_axis_external_field,
    sample_mirror_boundary_external_field,
    two_coil_on_axis_bz,
    write_mirror_free_boundary_circular_coil_scan,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("results/mirror/free_boundary_circular_coils"))
    parser.add_argument("--coil-radius", type=float, default=0.35)
    parser.add_argument("--separation", type=float, default=2.0)
    parser.add_argument("--current", type=float, default=1.0e6)
    parser.add_argument("--midplane-radius", type=float, default=0.3)
    parser.add_argument("--ns", type=int, default=7)
    parser.add_argument("--ntheta", type=int, default=32)
    parser.add_argument("--nxi", type=int, default=33)
    parser.add_argument("--n-segments", type=int, default=256)
    parser.add_argument("--betas", type=str, default="1,3,10")
    parser.add_argument("--pressure-scale-one-percent", type=float, default=1.0)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def _parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in str(value).replace(",", " ").split() if item.strip())


def _write_axis_plot(z, direct_bz, analytic_bz, *, outdir: Path) -> Path:
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(z, analytic_bz, "k-", linewidth=1.8, label="analytic two-coil")
    ax.plot(z, direct_bz, "o", markersize=4.0, label="direct-coil bridge")
    ax.set_xlabel("z")
    ax.set_ylabel("Bz on axis")
    ax.set_title("free-boundary circular-coil bridge")
    ax.legend(fontsize="small")
    fig.tight_layout()
    path = outdir / "free_boundary_circular_coils_axis_bz.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_boundary_bmag_plot(boundary_sample, *, outdir: Path) -> Path:
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    mesh = ax.pcolormesh(
        np.asarray(boundary_sample.z),
        np.asarray(boundary_sample.theta),
        np.asarray(boundary_sample.bmag),
        shading="auto",
    )
    ax.set_xlabel("z")
    ax.set_ylabel("theta")
    ax.set_title("external |B| on sampled mirror boundary")
    fig.colorbar(mesh, ax=ax, label="|B|")
    fig.tight_layout()
    path = outdir / "free_boundary_circular_coils_boundary_bmag.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_geometry_plot(grid, boundary, coils: MirrorCircularCoils, *, outdir: Path) -> Path:
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    theta = grid.theta
    radius = boundary.radius_on_grid_3d(grid)
    z = np.broadcast_to(grid.z[None, :], radius.shape)
    x = radius * np.cos(theta[:, None])
    y = radius * np.sin(theta[:, None])
    coil_theta = np.linspace(0.0, 2.0 * np.pi, 160)

    fig = plt.figure(figsize=(6.5, 4.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(z, x, y, color="lightgray", alpha=0.5, linewidth=0.0)
    for radius_m, z0 in zip(coils.radii_m, coils.z_centers_m, strict=True):
        ax.plot(
            np.full_like(coil_theta, z0),
            radius_m * np.cos(coil_theta),
            radius_m * np.sin(coil_theta),
            color="tab:orange",
            linewidth=2.0,
        )
    ax.set_xlabel("z")
    ax.set_ylabel("x")
    ax.set_zlabel("y")
    ax.set_title("mirror boundary and circular coils")
    ax.set_box_aspect([max(1.0, float(np.ptp(grid.z))), 1, 1])
    ax.view_init(elev=18, azim=-62)
    fig.tight_layout()
    path = outdir / "free_boundary_circular_coils_geometry.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def run_case(
    outdir: Path,
    *,
    coil_radius: float = 0.35,
    separation: float = 2.0,
    current: float = 1.0e6,
    midplane_radius: float = 0.3,
    ns: int = 7,
    ntheta: int = 32,
    nxi: int = 33,
    n_segments: int = 256,
    betas: tuple[float, ...] = (1.0, 3.0, 10.0),
    pressure_scale_one_percent: float = 1.0,
    write_plots: bool = True,
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    grid = make_mirror_grid(
        ns=ns, ntheta=ntheta, nxi=nxi, mpol=max(0, (ntheta - 1) // 2), z_min=-0.5 * separation, z_max=0.5 * separation
    )
    coils = MirrorCircularCoils.symmetric_pair(
        coil_radius_m=coil_radius,
        separation_m=separation,
        current_a=current,
        n_segments=n_segments,
    )
    scan = make_mirror_free_boundary_circular_coil_scan(
        coils,
        betas,
        pressure_scale_for_one_percent=pressure_scale_one_percent,
    )
    analytic_bz = two_coil_on_axis_bz(
        grid.z,
        coil_radius_m=coil_radius,
        separation_m=separation,
        current_a=current,
    )
    boundary = initial_mirror_boundary_from_circular_coil_scan(
        grid,
        scan,
        midplane_radius=midplane_radius,
    )
    axis_sample = sample_mirror_axis_external_field(grid, coils)
    boundary_sample = sample_mirror_boundary_external_field(grid, boundary, coils)
    direct_bz = np.asarray(axis_sample.bz, dtype=float)
    relative_error = np.max(np.abs(direct_bz - analytic_bz) / np.maximum(np.abs(analytic_bz), np.finfo(float).tiny))
    setup_path = write_mirror_free_boundary_circular_coil_scan(
        outdir / "free_boundary_circular_coils_setup.json",
        scan,
    )

    figure_paths: dict[str, str] = {}
    if write_plots:
        figure_dir = outdir / "figures"
        figure_paths["axis_bz"] = str(_write_axis_plot(grid.z, direct_bz, analytic_bz, outdir=figure_dir))
        figure_paths["boundary_bmag"] = str(_write_boundary_bmag_plot(boundary_sample, outdir=figure_dir))
        figure_paths["geometry"] = str(_write_geometry_plot(grid, boundary, coils, outdir=figure_dir))

    metrics = {
        "coil_radius": float(coil_radius),
        "separation": float(separation),
        "current": float(current),
        "midplane_radius": float(midplane_radius),
        "ns": int(ns),
        "ntheta": int(ntheta),
        "nxi": int(nxi),
        "n_segments": int(n_segments),
        "axis_bz_relative_linf": float(relative_error),
        "axis_bz_min": float(np.min(np.abs(direct_bz))),
        "axis_bz_max": float(np.max(np.abs(direct_bz))),
        "boundary_bmag_min": float(np.min(np.asarray(boundary_sample.bmag))),
        "boundary_bmag_max": float(np.max(np.asarray(boundary_sample.bmag))),
        "setup_json": str(setup_path),
        "beta_cases": [case.to_dict() for case in scan.beta_cases],
        "figures": figure_paths,
    }
    metrics_path = outdir / "free_boundary_circular_coils_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics_path


def main() -> None:
    args = build_parser().parse_args()
    path = run_case(
        args.outdir,
        coil_radius=args.coil_radius,
        separation=args.separation,
        current=args.current,
        midplane_radius=args.midplane_radius,
        ns=args.ns,
        ntheta=args.ntheta,
        nxi=args.nxi,
        n_segments=args.n_segments,
        betas=_parse_float_list(args.betas),
        pressure_scale_one_percent=args.pressure_scale_one_percent,
        write_plots=not args.no_plots,
    )
    print(path)


if __name__ == "__main__":
    main()
