"""Run a finite-current mirror example with visibly pitched field lines."""

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
    two_coil_on_axis_bz,
    write_mirror_output,
)
from vmec_jax.mirror.plotting.bfield import mirror_boundary_field_line_data
from vmec_jax.mirror.plotting.diagnostics import mirror_boozer_like_summary_metrics
from vmec_jax.mirror.plotting.geometry import mirror_boundary_3d_data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("results/mirror/finite_current_pitch"))
    parser.add_argument("--coil-radius", type=float, default=0.35)
    parser.add_argument("--separation", type=float, default=2.0)
    parser.add_argument("--coil-current", type=float, default=1.0e6)
    parser.add_argument("--midplane-radius", type=float, default=0.3)
    parser.add_argument("--i-prime", type=float, default=0.01)
    parser.add_argument("--ns", type=int, default=9)
    parser.add_argument("--nxi", type=int, default=33)
    parser.add_argument("--maxiter", type=int, default=4)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def _pitch_metrics(output, analytic_bz) -> dict[str, float]:
    lines = mirror_boundary_field_line_data(output, num_lines=6)
    theta_advance = np.asarray(lines.theta[:, -1] - lines.theta[:, 0], dtype=float)
    mirror_bz = np.asarray(output.field.b_z[0, 0], dtype=float)
    analytic_bz = np.asarray(analytic_bz, dtype=float)
    relative_error = np.abs(mirror_bz - analytic_bz) / np.maximum(np.abs(analytic_bz), np.finfo(float).tiny)
    with np.errstate(divide="ignore", invalid="ignore"):
        twist = np.divide(
            output.profiles.i_prime,
            output.profiles.psi_prime,
            out=np.zeros_like(output.profiles.i_prime),
            where=np.abs(output.profiles.psi_prime) > 0.0,
        )
    metrics = {
        "axis_bz_relative_linf": float(np.max(relative_error)),
        "analytic_mirror_ratio": on_axis_mirror_ratio(analytic_bz),
        "mirror_axis_mirror_ratio": on_axis_mirror_ratio(mirror_bz),
        "mirror_output_mirror_ratio": float(output.diagnostics.mirror_ratio),
        "field_line_theta_advance_mean": float(np.mean(theta_advance)),
        "field_line_theta_advance_min": float(np.min(theta_advance)),
        "field_line_theta_advance_max": float(np.max(theta_advance)),
        "field_line_turns_mean": float(np.mean(theta_advance) / (2.0 * np.pi)),
        "twist_proxy_i_prime_over_psi_prime_mean": float(np.mean(twist)),
        "final_residual_norm": float(output.diagnostics.residual_norm),
        "final_fsq": float(output.diagnostics.fsq),
        "final_normalized_force": float(output.diagnostics.normalized_force),
        "active_force_dof": int(output.diagnostics.active_force_dof),
        "final_energy_total": float(output.diagnostics.energy_total),
        "min_sqrtg": float(output.diagnostics.min_sqrtg),
    }
    metrics.update(mirror_boozer_like_summary_metrics(output))
    return metrics


def _write_pitch_plot(output, *, outdir: Path) -> Path:
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    lines = mirror_boundary_field_line_data(output, num_lines=6)
    fig, ax = plt.subplots(figsize=(6.5, 3.75))
    for line_index in range(lines.theta.shape[0]):
        ax.plot(lines.z[line_index], lines.theta[line_index] - lines.theta[line_index, 0], linewidth=1.5)
    ax.set_xlabel("z")
    ax.set_ylabel("theta advance")
    ax.set_title("boundary field-line pitch")
    fig.tight_layout()
    path = outdir / "finite_current_pitch_theta_advance.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_geometry_coils_and_lines_plot(output, *, coil_radius: float, separation: float, outdir: Path) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize

    outdir.mkdir(parents=True, exist_ok=True)
    boundary = mirror_boundary_3d_data(output)
    lines = mirror_boundary_field_line_data(output, num_lines=6)
    norm = Normalize(vmin=float(np.min(boundary.bmag)), vmax=float(np.max(boundary.bmag)))
    theta = np.linspace(0.0, 2.0 * np.pi, 128)

    fig = plt.figure(figsize=(6.75, 4.75))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        boundary.z,
        boundary.x,
        boundary.y,
        facecolors=cm.viridis(norm(boundary.bmag)),
        linewidth=0.0,
        alpha=0.82,
    )
    for z0 in (-0.5 * float(separation), 0.5 * float(separation)):
        z = np.full_like(theta, z0)
        x = float(coil_radius) * np.cos(theta)
        y = float(coil_radius) * np.sin(theta)
        ax.plot(z, x, y, color="tab:orange", linewidth=2.0)
    line_scale = 1.025
    for line_index in range(lines.z.shape[0]):
        ax.plot(
            lines.z[line_index],
            line_scale * lines.x[line_index],
            line_scale * lines.y[line_index],
            color="white",
            linewidth=2.5,
        )
        ax.plot(
            lines.z[line_index],
            line_scale * lines.x[line_index],
            line_scale * lines.y[line_index],
            color="tab:red",
            linewidth=1.5,
        )
    ax.set_xlabel("z")
    ax.set_ylabel("x")
    ax.set_zlabel("y")
    ax.set_title("finite-current mirror pitch")
    ax.set_box_aspect([max(1.0, float(np.ptp(output.z))), 1, 1])
    ax.view_init(elev=18, azim=-62)
    mappable = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
    mappable.set_array([])
    fig.colorbar(mappable, ax=ax, shrink=0.65, pad=0.08, label="|B|")
    fig.tight_layout()
    path = outdir / "finite_current_pitch_geometry_coils_field_lines.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def run_case(
    outdir: Path,
    *,
    coil_radius: float = 0.35,
    separation: float = 2.0,
    coil_current: float = 1.0e6,
    midplane_radius: float = 0.3,
    i_prime: float = 0.01,
    ns: int = 9,
    nxi: int = 33,
    maxiter: int = 4,
    write_plots: bool = True,
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
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
        current_a=coil_current,
    )
    midplane_bz = float(
        two_coil_on_axis_bz(
            0.0,
            coil_radius_m=coil_radius,
            separation_m=separation,
            current_a=coil_current,
        )
    )
    psi_value = 0.5 * abs(midplane_bz) * float(midplane_radius) ** 2
    boundary = mirror_boundary_from_on_axis_bz(psi_value, grid.z, analytic_bz)
    result = run_mirror_fixed_boundary(
        config,
        boundary,
        psi_prime=PsiPrimeProfile.constant(psi_value),
        i_prime=IPrimeProfile.constant(i_prime),
        pressure=PressureProfile.zero(),
        options=MirrorSolveOptions(optimizer="lbfgs", maxiter=maxiter, tolerance=1.0e-10, mu0=1.0),
    )
    mout = write_mirror_output(outdir / "mout_finite_current_pitch.nc", result, overwrite=True)
    output = load_mirror_output(mout)
    metrics = _pitch_metrics(output, analytic_bz)
    (outdir / "finite_current_pitch_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    if write_plots:
        figure_dir = outdir / "figures"
        plot_mirror_output(mout, outdir=figure_dir)
        _write_pitch_plot(output, outdir=figure_dir)
        _write_geometry_coils_and_lines_plot(output, coil_radius=coil_radius, separation=separation, outdir=figure_dir)
    return mout


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    mout = run_case(
        args.outdir,
        coil_radius=args.coil_radius,
        separation=args.separation,
        coil_current=args.coil_current,
        midplane_radius=args.midplane_radius,
        i_prime=args.i_prime,
        ns=args.ns,
        nxi=args.nxi,
        maxiter=args.maxiter,
        write_plots=not args.no_plots,
    )
    print(mout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
