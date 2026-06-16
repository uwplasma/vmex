"""Run an axisymmetric mirror benchmark from an analytic two-coil field."""

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
    IPrimeProfile,
    MirrorConfig,
    MirrorResolution,
    MirrorSolveOptions,
    PressureProfile,
    PsiPrimeProfile,
    load_mirror_output,
    mirror_boundary_from_on_axis_bz,
    on_axis_mirror_ratio,
    plot_mirror_output,
    run_mirror_fixed_boundary,
    two_coil_field_rz,
    two_coil_on_axis_bz,
    write_mirror_output,
)
from vmec_jax.mirror.plotting.geometry import mirror_boundary_3d_data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("results/mirror/two_coil_axisym"))
    parser.add_argument("--coil-radius", type=float, default=0.35)
    parser.add_argument("--separation", type=float, default=2.0)
    parser.add_argument("--current", type=float, default=1.0e6)
    parser.add_argument("--midplane-radius", type=float, default=0.3)
    parser.add_argument("--ns", type=int, default=9)
    parser.add_argument("--nxi", type=int, default=33)
    parser.add_argument("--maxiter", type=int, default=8)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def _axis_comparison_metrics(output, analytic_bz) -> dict[str, float]:
    mirror_bz = np.asarray(output.field.b_z[0, 0], dtype=float)
    analytic_bz = np.asarray(analytic_bz, dtype=float)
    relative_error = np.abs(mirror_bz - analytic_bz) / np.maximum(np.abs(analytic_bz), np.finfo(float).tiny)
    return {
        "axis_bz_relative_linf": float(np.max(relative_error)),
        "analytic_mirror_ratio": on_axis_mirror_ratio(analytic_bz),
        "mirror_axis_mirror_ratio": on_axis_mirror_ratio(mirror_bz),
        "mirror_output_mirror_ratio": float(output.diagnostics.mirror_ratio),
        "final_residual_norm": float(output.diagnostics.residual_norm),
        "final_fsq": float(output.diagnostics.fsq),
        "final_normalized_force": float(output.diagnostics.normalized_force),
        "active_force_dof": int(output.diagnostics.active_force_dof),
        "final_energy_total": float(output.diagnostics.energy_total),
        "min_sqrtg": float(output.diagnostics.min_sqrtg),
    }


def _write_axis_comparison_plot(output, analytic_bz, *, outdir: Path) -> Path:
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    z = np.asarray(output.z, dtype=float)
    mirror_bz = np.asarray(output.field.b_z[0, 0], dtype=float)
    fig, ax = plt.subplots(figsize=(6.5, 3.75))
    ax.plot(z, analytic_bz, "k-", linewidth=1.8, label="two-coil analytic")
    ax.plot(z, mirror_bz, "o", markersize=4.0, label="mirror on-axis")
    ax.set_xlabel("z")
    ax.set_ylabel("Bz on axis")
    ax.set_title("two-coil on-axis benchmark")
    ax.legend(fontsize="small")
    fig.tight_layout()
    path = outdir / "two_coil_axisym_axis_bz_comparison.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_geometry_with_coils_plot(output, *, coil_radius: float, separation: float, outdir: Path) -> Path:
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    theta = np.linspace(0.0, 2.0 * np.pi, 128)
    boundary = mirror_boundary_3d_data(output)
    fig = plt.figure(figsize=(6.5, 4.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        boundary.z,
        boundary.x,
        boundary.y,
        color="lightgray",
        alpha=0.45,
        linewidth=0.0,
    )
    for z0 in (-0.5 * float(separation), 0.5 * float(separation)):
        z = np.full_like(theta, z0)
        x = float(coil_radius) * np.cos(theta)
        y = float(coil_radius) * np.sin(theta)
        ax.plot(z, x, y, color="tab:orange", linewidth=2.0)
    ax.set_xlabel("z")
    ax.set_ylabel("x")
    ax.set_zlabel("y")
    ax.set_title("two-coil mirror flux tube")
    ax.set_box_aspect([max(1.0, float(np.ptp(output.z))), 1, 1])
    ax.view_init(elev=18, azim=-62)
    fig.tight_layout()
    path = outdir / "two_coil_axisym_geometry_with_coils.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_bmag_with_coils_plot(output, *, coil_radius: float, separation: float, outdir: Path) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize

    outdir.mkdir(parents=True, exist_ok=True)
    theta = np.linspace(0.0, 2.0 * np.pi, 128)
    boundary = mirror_boundary_3d_data(output)
    norm = Normalize(vmin=float(np.min(boundary.bmag)), vmax=float(np.max(boundary.bmag)))
    fig = plt.figure(figsize=(6.75, 4.75))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        boundary.z,
        boundary.x,
        boundary.y,
        facecolors=cm.viridis(norm(boundary.bmag)),
        linewidth=0.0,
        alpha=0.9,
    )
    for z0 in (-0.5 * float(separation), 0.5 * float(separation)):
        z = np.full_like(theta, z0)
        x = float(coil_radius) * np.cos(theta)
        y = float(coil_radius) * np.sin(theta)
        ax.plot(z, x, y, color="tab:orange", linewidth=2.0)
    ax.set_xlabel("z")
    ax.set_ylabel("x")
    ax.set_zlabel("y")
    ax.set_title("two-coil mirror |B|")
    ax.set_box_aspect([max(1.0, float(np.ptp(output.z))), 1, 1])
    ax.view_init(elev=18, azim=-62)
    mappable = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
    mappable.set_array([])
    fig.colorbar(mappable, ax=ax, shrink=0.65, pad=0.08, label="|B|")
    fig.tight_layout()
    path = outdir / "two_coil_axisym_bmag_with_coils.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _off_axis_comparison(output, *, coil_radius: float, separation: float, current: float, surface_index: int = 1):
    surface_index = min(max(1, int(surface_index)), output.ns - 1)
    z = np.asarray(output.z, dtype=float)
    radius = np.asarray(output.geometry.r[surface_index, 0], dtype=float)
    analytic = two_coil_field_rz(radius, z, coil_radius_m=coil_radius, separation_m=separation, current_a=current)
    mirror_br = np.asarray(output.field.b_x[surface_index, 0], dtype=float)
    mirror_bz = np.asarray(output.field.b_z[surface_index, 0], dtype=float)
    br_error = np.abs(mirror_br - analytic.br)
    bz_error = np.abs(mirror_bz - analytic.bz)
    br_scale = np.maximum(np.abs(analytic.br), 1.0e-12)
    bz_scale = np.maximum(np.abs(analytic.bz), 1.0e-12)
    metrics = {
        "surface_index": surface_index,
        "surface_s": float(output.s[surface_index]),
        "max_radius": float(np.max(radius)),
        "br_relative_linf": float(np.max(br_error / br_scale)),
        "bz_relative_linf": float(np.max(bz_error / bz_scale)),
        "br_absolute_linf": float(np.max(br_error)),
        "bz_absolute_linf": float(np.max(bz_error)),
    }
    return radius, analytic, mirror_br, mirror_bz, metrics


def _write_off_axis_comparison_plot(
    output,
    *,
    coil_radius: float,
    separation: float,
    current: float,
    outdir: Path,
) -> tuple[Path, dict[str, float]]:
    import matplotlib.pyplot as plt

    radius, analytic, mirror_br, mirror_bz, metrics = _off_axis_comparison(
        output,
        coil_radius=coil_radius,
        separation=separation,
        current=current,
    )
    del radius
    outdir.mkdir(parents=True, exist_ok=True)
    z = np.asarray(output.z, dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 5.0), sharex=True)
    axes[0].plot(z, analytic.br, "k-", linewidth=1.8, label="Biot-Savart Br")
    axes[0].plot(z, mirror_br, "o", markersize=4.0, label="mirror Br")
    axes[0].set_ylabel("Br")
    axes[0].legend(fontsize="small")
    axes[1].plot(z, analytic.bz, "k-", linewidth=1.8, label="Biot-Savart Bz")
    axes[1].plot(z, mirror_bz, "o", markersize=4.0, label="mirror Bz")
    axes[1].set_xlabel("z")
    axes[1].set_ylabel("Bz")
    axes[1].legend(fontsize="small")
    fig.suptitle(f"off-axis low-radius comparison, s={metrics['surface_s']:.3g}")
    fig.tight_layout()
    path = outdir / "two_coil_axisym_off_axis_biot_savart_comparison.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path, metrics


def _run_one_case(
    outdir: Path,
    *,
    coil_radius: float,
    separation: float,
    current: float,
    midplane_radius: float,
    ns: int,
    nxi: int,
    maxiter: int,
) -> tuple[Path, np.ndarray]:
    half_separation = 0.5 * float(separation)
    config = MirrorConfig(
        MirrorResolution(ns=int(ns), ntheta=1, nxi=int(nxi), mpol=0),
        z_min=-half_separation,
        z_max=half_separation,
    )
    grid = config.build_grid()
    analytic_bz = two_coil_on_axis_bz(
        grid.z,
        coil_radius_m=coil_radius,
        separation_m=separation,
        current_a=current,
    )
    midplane_bz = float(two_coil_on_axis_bz(0.0, coil_radius_m=coil_radius, separation_m=separation, current_a=current))
    psi_value = 0.5 * abs(midplane_bz) * float(midplane_radius) ** 2
    boundary = mirror_boundary_from_on_axis_bz(psi_value, grid.z, analytic_bz)
    result = run_mirror_fixed_boundary(
        config,
        boundary,
        psi_prime=PsiPrimeProfile.constant(psi_value),
        i_prime=IPrimeProfile.zero(),
        pressure=PressureProfile.zero(),
        options=MirrorSolveOptions(optimizer="lbfgs", maxiter=maxiter, tolerance=1.0e-10, mu0=1.0),
    )
    mout = write_mirror_output(outdir / f"mout_two_coil_axisym_ns{int(ns)}_nxi{int(nxi)}.nc", result, overwrite=True)
    return mout, analytic_bz


def _run_convergence_study(
    outdir: Path,
    *,
    coil_radius: float,
    separation: float,
    current: float,
    midplane_radius: float,
    maxiter: int,
) -> list[dict[str, float]]:
    grid_cases = [(7, 17), (9, 25), (11, 33)]
    rows = []
    study_dir = outdir / "convergence"
    study_dir.mkdir(parents=True, exist_ok=True)
    for ns, nxi in grid_cases:
        mout, analytic_bz = _run_one_case(
            study_dir,
            coil_radius=coil_radius,
            separation=separation,
            current=current,
            midplane_radius=midplane_radius,
            ns=ns,
            nxi=nxi,
            maxiter=maxiter,
        )
        output = load_mirror_output(mout)
        metrics = _axis_comparison_metrics(output, analytic_bz)
        _, _, _, _, off_axis_metrics = _off_axis_comparison(
            output,
            coil_radius=coil_radius,
            separation=separation,
            current=current,
        )
        rows.append(
            {
                "ns": int(ns),
                "nxi": int(nxi),
                "axis_bz_relative_linf": metrics["axis_bz_relative_linf"],
                "off_axis_br_relative_linf": off_axis_metrics["br_relative_linf"],
                "off_axis_bz_relative_linf": off_axis_metrics["bz_relative_linf"],
                "mirror_ratio_error": abs(metrics["mirror_axis_mirror_ratio"] - metrics["analytic_mirror_ratio"]),
                "fsq": metrics["final_fsq"],
                "normalized_force": metrics["final_normalized_force"],
                "active_force_dof": metrics["active_force_dof"],
                "min_sqrtg": metrics["min_sqrtg"],
            }
        )
    (outdir / "two_coil_axisym_convergence.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    return rows


def _write_convergence_plot(rows: list[dict[str, float]], *, outdir: Path) -> Path:
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    labels = [f"{row['ns']}/{row['nxi']}" for row in rows]
    x = np.arange(len(rows))
    axis_error = np.asarray([row["axis_bz_relative_linf"] for row in rows], dtype=float)
    br_error = np.asarray([row["off_axis_br_relative_linf"] for row in rows], dtype=float)
    bz_error = np.asarray([row["off_axis_bz_relative_linf"] for row in rows], dtype=float)
    ratio_error = np.asarray([row["mirror_ratio_error"] for row in rows], dtype=float)
    fsq = np.asarray([row["fsq"] for row in rows], dtype=float)
    normalized_force = np.asarray([row["normalized_force"] for row in rows], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(6.5, 5.0), sharex=True)
    axes[0].semilogy(x, np.maximum(axis_error, 1.0e-300), "o-", label="axis Bz rel error")
    axes[0].semilogy(x, np.maximum(br_error, 1.0e-300), "o-", label="low-radius Br rel error")
    axes[0].semilogy(x, np.maximum(bz_error, 1.0e-300), "o-", label="low-radius Bz rel error")
    axes[0].semilogy(x, np.maximum(ratio_error, 1.0e-300), "o-", label="mirror-ratio abs error")
    axes[0].set_ylabel("field error")
    axes[0].legend(fontsize="small")
    axes[1].semilogy(x, np.maximum(fsq, 1.0e-300), "o-", label="mirror fsq")
    axes[1].semilogy(x, np.maximum(normalized_force, 1.0e-300), "o-", label="normalized force")
    axes[1].set_ylabel("force metric")
    axes[1].set_xlabel("ns/nxi")
    axes[1].set_xticks(x, labels)
    axes[1].legend(fontsize="small")
    fig.suptitle("two-coil benchmark convergence")
    fig.tight_layout()
    path = outdir / "two_coil_axisym_convergence.png"
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
    ns: int = 9,
    nxi: int = 33,
    maxiter: int = 8,
    write_plots: bool = True,
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    mout, analytic_bz = _run_one_case(
        outdir,
        coil_radius=coil_radius,
        separation=separation,
        current=current,
        midplane_radius=midplane_radius,
        ns=ns,
        nxi=nxi,
        maxiter=maxiter,
    )
    final_mout = outdir / "mout_two_coil_axisym.nc"
    Path(mout).replace(final_mout)
    mout = final_mout
    output = load_mirror_output(mout)
    metrics = _axis_comparison_metrics(output, analytic_bz)
    _, _, _, _, off_axis_metrics = _off_axis_comparison(
        output,
        coil_radius=coil_radius,
        separation=separation,
        current=current,
    )
    metrics["off_axis_br_relative_linf"] = off_axis_metrics["br_relative_linf"]
    metrics["off_axis_bz_relative_linf"] = off_axis_metrics["bz_relative_linf"]
    metrics["off_axis_surface_s"] = off_axis_metrics["surface_s"]
    (outdir / "two_coil_axisym_off_axis_metrics.json").write_text(
        json.dumps(off_axis_metrics, indent=2, sort_keys=True) + "\n"
    )
    if write_plots:
        figure_dir = outdir / "figures"
        plot_mirror_output(mout, outdir=figure_dir)
        _write_axis_comparison_plot(output, analytic_bz, outdir=figure_dir)
        _write_geometry_with_coils_plot(output, coil_radius=coil_radius, separation=separation, outdir=figure_dir)
        _write_bmag_with_coils_plot(output, coil_radius=coil_radius, separation=separation, outdir=figure_dir)
        _, off_axis_metrics = _write_off_axis_comparison_plot(
            output,
            coil_radius=coil_radius,
            separation=separation,
            current=current,
            outdir=figure_dir,
        )
        convergence_rows = _run_convergence_study(
            outdir,
            coil_radius=coil_radius,
            separation=separation,
            current=current,
            midplane_radius=midplane_radius,
            maxiter=0,
        )
        _write_convergence_plot(convergence_rows, outdir=figure_dir)
    (outdir / "two_coil_axisym_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return mout


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    mout = run_case(
        args.outdir,
        coil_radius=args.coil_radius,
        separation=args.separation,
        current=args.current,
        midplane_radius=args.midplane_radius,
        ns=args.ns,
        nxi=args.nxi,
        maxiter=args.maxiter,
        write_plots=not args.no_plots,
    )
    print(mout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
