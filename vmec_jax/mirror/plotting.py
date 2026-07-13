"""Diagnostic plots for mirror-native ``mout`` equilibrium files."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .output import read_mout

_DPI = 110


def _matplotlib():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _as_mout(mout):
    if hasattr(mout, "boundary_radius") and hasattr(mout, "b_xyz"):
        return mout, "mout"
    path = Path(mout)
    stem = path.stem
    return read_mout(path), stem[5:] if stem.startswith("mout_") else stem


def _theta_samples(data, values, theta_dense):
    """Periodically interpolate a ``(ntheta, nxi)`` mirror table."""

    values = np.asarray(values, dtype=float)
    theta = np.asarray(data.theta, dtype=float)
    if theta.size == 1:
        return np.broadcast_to(values[0], (len(theta_dense), values.shape[1]))
    order = np.argsort(np.mod(theta, 2.0 * np.pi))
    theta = np.mod(theta[order], 2.0 * np.pi)
    table = values[order]
    theta_extended = np.concatenate([theta, [theta[0] + 2.0 * np.pi]])
    table_extended = np.concatenate([table, table[:1]], axis=0)
    return np.stack([
        np.interp(np.mod(theta_dense, 2.0 * np.pi), theta_extended, table_extended[:, iz])
        for iz in range(values.shape[1])
    ], axis=1)


def _field_line(data, radial_index: int, theta0: float, z_order):
    """Trace one cap-to-cap line using the saved Cartesian field samples."""

    z = np.asarray(data.z, dtype=float)[z_order]
    theta_nodes = np.asarray(data.theta, dtype=float)
    b_xyz = np.take(np.asarray(data.b_xyz, dtype=float)[radial_index], z_order, axis=1)
    radius = np.sqrt(float(np.asarray(data.s)[radial_index])) * np.take(
        np.asarray(data.radius_scale)[radial_index], z_order, axis=1
    )
    angles = np.empty(z.size)
    angles[0] = theta0
    periodic_theta = np.r_[np.mod(theta_nodes, 2.0 * np.pi), 2.0 * np.pi]
    for iz in range(z.size - 1):
        angle = angles[iz]
        if theta_nodes.size == 1:
            vector = b_xyz[0, iz]
            local_radius = radius[0, iz]
        else:
            vector = np.asarray([
                np.interp(
                    np.mod(angle, 2.0 * np.pi),
                    periodic_theta,
                    np.r_[b_xyz[:, iz, component], b_xyz[0, iz, component]],
                )
                for component in range(3)
            ])
            local_radius = np.interp(
                np.mod(angle, 2.0 * np.pi),
                periodic_theta,
                np.r_[radius[:, iz], radius[0, iz]],
            )
        b_theta = -np.sin(angle) * vector[0] + np.cos(angle) * vector[1]
        denominator = local_radius * vector[2]
        pitch = 0.0 if abs(denominator) < 1.0e-14 else b_theta / denominator
        angles[iz + 1] = angle + (z[iz + 1] - z[iz]) * pitch
    radius_table = np.take(np.asarray(data.radius_scale)[radial_index], z_order, axis=1)
    radius_samples = _theta_samples(data, radius_table, angles)
    radius_line = radius_samples[np.arange(z.size), np.arange(z.size)] * np.sqrt(
        float(np.asarray(data.s)[radial_index])
    )
    return z, radius_line * np.cos(angles), radius_line * np.sin(angles)


def plot_mout(mout, outdir: str | Path, *, name: str | None = None) -> dict[str, Path]:
    """Render summary, cross-section, ``|B|``, and horizontal 3D mirror plots."""

    plt = _matplotlib()
    data, default_name = _as_mout(mout)
    label = name or default_name
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    z_order = np.argsort(np.asarray(data.z))
    z = np.asarray(data.z)[z_order]
    s = np.asarray(data.s)
    center = int(np.argmin(np.abs(z)))
    boundary = np.take(np.asarray(data.boundary_radius), z_order, axis=1)
    mod_b = np.take(np.asarray(data.mod_b), z_order, axis=2)
    pressure = np.take(np.asarray(data.p_perpendicular), z_order, axis=2)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), constrained_layout=True)
    axes[0, 0].plot(z, np.mean(boundary, axis=0), color="#0072B2", lw=2)
    axes[0, 0].fill_between(
        z, np.min(boundary, axis=0), np.max(boundary, axis=0),
        color="#0072B2", alpha=0.2,
    )
    axes[0, 0].set(title="Solved LCFS", xlabel="Axial position z [m]", ylabel="Radius [m]")
    axes[0, 1].plot(z, np.mean(mod_b[0], axis=0), label="axis", color="#009E73", lw=2)
    axes[0, 1].plot(z, np.mean(mod_b[-1], axis=0), label="LCFS", color="#D55E00", lw=2)
    axes[0, 1].set(title="Magnetic-field strength", xlabel="Axial position z [m]", ylabel="|B| [T]")
    axes[0, 1].legend()
    axes[1, 0].plot(
        np.sqrt(s), np.mean(pressure[:, :, center], axis=1) / 1.0e3,
        "o-", color="#CC79A7",
    )
    axes[1, 0].set(title="Midplane pressure", xlabel="Normalized radius sqrt(s)", ylabel="p_perp [kPa]")
    history = np.asarray(data.history)
    if history.size:
        axes[1, 1].semilogy(history[:, 0], np.maximum(history[:, -1], 1.0e-18), color="#0072B2")
    axes[1, 1].axhline(float(data.ftol), color="0.25", ls="--", lw=1, label="ftol")
    axes[1, 1].set(
        title=f"Convergence ({int(data.iterations)} iterations)",
        xlabel="Residual evaluation", ylabel="Maximum normalized residual",
    )
    axes[1, 1].legend()
    for ax in axes.flat:
        ax.grid(alpha=0.22)
    paths["summary"] = outdir / f"{label}_summary.png"
    fig.savefig(paths["summary"], dpi=_DPI, bbox_inches="tight")
    plt.close(fig)

    theta_dense = np.linspace(0.0, 2.0 * np.pi, 129)
    indices = np.unique(np.round(np.linspace(0, len(z) - 1, 6)).astype(int))
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.8), constrained_layout=True)
    for ax, iz in zip(axes.flat, indices, strict=False):
        for radial_index in np.unique(np.round(np.linspace(0, len(s) - 1, 7)).astype(int)):
            table = np.take(np.asarray(data.radius_scale)[radial_index], z_order, axis=1)
            radius = np.sqrt(s[radial_index]) * _theta_samples(data, table, theta_dense)[:, iz]
            ax.plot(radius * np.cos(theta_dense), radius * np.sin(theta_dense), lw=0.9)
        ax.set(title=f"z = {z[iz]:.3g} m", xlabel="x [m]", ylabel="y [m]", aspect="equal")
    for ax in axes.flat[len(indices):]:
        ax.set_visible(False)
    paths["cross_sections"] = outdir / f"{label}_cross_sections.png"
    fig.savefig(paths["cross_sections"], dpi=_DPI, bbox_inches="tight")
    plt.close(fig)

    boundary_b = _theta_samples(data, mod_b[-1], theta_dense)
    fig, ax = plt.subplots(figsize=(10.5, 4.2), constrained_layout=True)
    contour = ax.contour(z, theta_dense, boundary_b, 18, cmap="viridis", linewidths=0.9)
    ax.clabel(contour, inline=True, fontsize=7, fmt="%.3g")
    fig.colorbar(contour, ax=ax, label="LCFS |B| [T]")
    ax.set(title="Boundary magnetic-field strength", xlabel="Axial position z [m]", ylabel="Poloidal angle theta")
    paths["modB"] = outdir / f"{label}_modB.png"
    fig.savefig(paths["modB"], dpi=_DPI, bbox_inches="tight")
    plt.close(fig)

    radius_dense = _theta_samples(data, boundary, theta_dense)
    zz, tt = np.meshgrid(z, theta_dense)
    fig = plt.figure(figsize=(11.5, 6.2), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    norm = plt.Normalize(float(np.min(boundary_b)), float(np.max(boundary_b)))
    surface = ax.plot_surface(
        zz, radius_dense * np.cos(tt), radius_dense * np.sin(tt),
        facecolors=plt.cm.viridis(norm(boundary_b)), linewidth=0, alpha=0.8,
    )
    surface.set_rasterized(True)
    for coil in np.asarray(data.coil_xyz):
        closed = np.vstack([coil, coil[0]])
        ax.plot(closed[:, 2], closed[:, 0], closed[:, 1], color="#C44E52", lw=2)
    for theta0 in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
        line_z, line_x, line_y = _field_line(data, len(s) - 1, theta0, z_order)
        ax.plot(line_z, 1.01 * line_x, 1.01 * line_y, color="black", lw=3.2)
        ax.plot(line_z, 1.01 * line_x, 1.01 * line_y, color="#00BFC4", lw=1.5)
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax,
        shrink=0.72, pad=0.05, label="LCFS |B| [T]",
    )
    ax.set(title="Solved mirror equilibrium", xlabel="z [m]", ylabel="x [m]", zlabel="y [m]")
    ax.set_box_aspect((2.2, 1.0, 1.0))
    ax.view_init(elev=22, azim=-57)
    paths["3d"] = outdir / f"{label}_3d.png"
    fig.savefig(paths["3d"], dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return paths


__all__ = ["plot_mout"]
