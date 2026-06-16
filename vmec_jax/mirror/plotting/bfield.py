"""Magnetic-field plot-data helpers for mirror output files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..io.mout import load_mirror_output
from ..io.schema import MirrorOutput
from .geometry import _import_matplotlib, _plot_name


@dataclass(frozen=True)
class MirrorBmagSXiData:
    """Theta-averaged ``|B|(s, xi)`` data."""

    s: np.ndarray
    xi: np.ndarray
    bmag: np.ndarray


@dataclass(frozen=True)
class MirrorBmagBoundaryData:
    """Boundary ``|B|(theta, xi)`` data."""

    theta: np.ndarray
    xi: np.ndarray
    bmag: np.ndarray


@dataclass(frozen=True)
class MirrorBfieldBoundaryData:
    """Boundary magnetic-field vector data for 3-D plotting."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    bx: np.ndarray
    by: np.ndarray
    bz: np.ndarray
    bmag: np.ndarray


@dataclass(frozen=True)
class MirrorBoundaryFieldLineData:
    """Boundary field-line traces in physical coordinates."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    theta: np.ndarray


def _as_output(output_or_path) -> MirrorOutput:
    return output_or_path if isinstance(output_or_path, MirrorOutput) else load_mirror_output(output_or_path)


def mirror_bmag_sxi_data(output_or_path) -> MirrorBmagSXiData:
    """Return theta-averaged ``|B|`` over ``(s, xi)``."""
    output = _as_output(output_or_path)
    return MirrorBmagSXiData(
        s=np.asarray(output.s),
        xi=np.asarray(output.xi),
        bmag=np.mean(np.asarray(output.field.bmag), axis=1),
    )


def mirror_bmag_boundary_data(output_or_path) -> MirrorBmagBoundaryData:
    """Return boundary ``|B|`` over ``(theta, xi)``."""
    output = _as_output(output_or_path)
    return MirrorBmagBoundaryData(
        theta=np.asarray(output.theta),
        xi=np.asarray(output.xi),
        bmag=np.asarray(output.field.bmag[-1]),
    )


def mirror_bfield_boundary_data(output_or_path, *, stride_theta: int = 2, stride_xi: int = 2) -> MirrorBfieldBoundaryData:
    """Return boundary magnetic-field vectors subsampled for 3-D quiver plots."""
    output = _as_output(output_or_path)
    stride_theta = max(1, int(stride_theta))
    stride_xi = max(1, int(stride_xi))
    theta_slice = slice(None, None, stride_theta)
    xi_slice = slice(None, None, stride_xi)
    return MirrorBfieldBoundaryData(
        x=np.asarray(output.geometry.x[-1, theta_slice, xi_slice]),
        y=np.asarray(output.geometry.y[-1, theta_slice, xi_slice]),
        z=np.asarray(output.geometry.z[-1, theta_slice, xi_slice]),
        bx=np.asarray(output.field.b_x[-1, theta_slice, xi_slice]),
        by=np.asarray(output.field.b_y[-1, theta_slice, xi_slice]),
        bz=np.asarray(output.field.b_z[-1, theta_slice, xi_slice]),
        bmag=np.asarray(output.field.bmag[-1, theta_slice, xi_slice]),
    )


def _interp_periodic(theta_nodes, values, theta_value: float) -> float:
    theta_nodes = np.asarray(theta_nodes, dtype=float)
    values = np.asarray(values, dtype=float)
    period = 2.0 * np.pi
    theta_wrapped = float(np.mod(theta_value, period))
    extended_theta = np.concatenate([theta_nodes, theta_nodes[:1] + period])
    extended_values = np.concatenate([values, values[:1]])
    return float(np.interp(theta_wrapped, extended_theta, extended_values))


def mirror_boundary_field_line_data(output_or_path, *, num_lines: int = 6) -> MirrorBoundaryFieldLineData:
    """Trace boundary field lines from one end cap to the other in ``(theta, xi)``."""
    output = _as_output(output_or_path)
    num_lines = max(1, int(num_lines))
    theta_nodes = np.asarray(output.theta, dtype=float)
    xi = np.asarray(output.xi, dtype=float)
    z = np.asarray(output.z, dtype=float)
    start_theta = np.linspace(0.0, 2.0 * np.pi, num_lines, endpoint=False)
    theta_lines = np.zeros((num_lines, output.nxi), dtype=float)
    theta_lines[:, 0] = start_theta
    btheta = np.asarray(output.field.b_sup_theta[-1], dtype=float)
    bxi = np.asarray(output.field.b_sup_xi[-1], dtype=float)
    for line_index in range(num_lines):
        for k in range(output.nxi - 1):
            numerator = _interp_periodic(theta_nodes, btheta[:, k], theta_lines[line_index, k])
            denominator = _interp_periodic(theta_nodes, bxi[:, k], theta_lines[line_index, k])
            slope = 0.0 if abs(denominator) <= np.finfo(float).tiny else numerator / denominator
            theta_lines[line_index, k + 1] = theta_lines[line_index, k] + slope * (xi[k + 1] - xi[k])

    radius = np.zeros_like(theta_lines)
    boundary_r = np.asarray(output.geometry.boundary_r, dtype=float)
    for line_index in range(num_lines):
        for k in range(output.nxi):
            radius[line_index, k] = _interp_periodic(theta_nodes, boundary_r[:, k], theta_lines[line_index, k])
    return MirrorBoundaryFieldLineData(
        x=radius * np.cos(theta_lines),
        y=radius * np.sin(theta_lines),
        z=np.broadcast_to(z[None, :], theta_lines.shape).copy(),
        theta=theta_lines,
    )


def write_mirror_bmag_sxi(output_or_path, *, outdir: str | Path, name: str | None = None) -> Path:
    """Write the theta-averaged ``|B|(s, xi)`` map."""
    output = _as_output(output_or_path)
    data = mirror_bmag_sxi_data(output)
    plt = _import_matplotlib()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    mesh = ax.pcolormesh(data.xi, data.s, data.bmag, shading="auto")
    ax.set_xlabel("xi")
    ax.set_ylabel("s")
    fig.colorbar(mesh, ax=ax, label="|B|")
    fig.tight_layout()
    path = outdir / f"{_plot_name(output, name)}_mirror_bmag_sxi.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def write_mirror_bmag_boundary(output_or_path, *, outdir: str | Path, name: str | None = None) -> Path:
    """Write the boundary ``|B|(theta, xi)`` map."""
    output = _as_output(output_or_path)
    data = mirror_bmag_boundary_data(output)
    plt = _import_matplotlib()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    if data.theta.size == 1:
        ax.plot(data.xi, data.bmag[0], ".-")
        ax.set_ylabel("|B| at boundary")
    else:
        mesh = ax.pcolormesh(data.xi, data.theta, data.bmag, shading="auto")
        fig.colorbar(mesh, ax=ax, label="|B|")
        ax.set_ylabel("theta")
    ax.set_xlabel("xi")
    fig.tight_layout()
    path = outdir / f"{_plot_name(output, name)}_mirror_bmag_boundary.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def write_mirror_bfield_boundary(output_or_path, *, outdir: str | Path, name: str | None = None) -> Path:
    """Write a 3-D boundary magnetic-field vector plot."""
    output = _as_output(output_or_path)
    data = mirror_bfield_boundary_data(output)
    lines = mirror_boundary_field_line_data(output)
    plt = _import_matplotlib()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    scale = np.maximum(data.bmag, np.finfo(float).tiny)
    fig = plt.figure(figsize=(6.25, 4.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        np.asarray(output.geometry.z[-1]),
        np.asarray(output.geometry.x[-1]),
        np.asarray(output.geometry.y[-1]),
        color="lightgray",
        alpha=0.25,
        linewidth=0.0,
    )
    ax.quiver(
        data.z,
        data.x,
        data.y,
        data.bz / scale,
        data.bx / scale,
        data.by / scale,
        length=0.14,
        normalize=False,
        color="tab:blue",
    )
    for line_index in range(lines.z.shape[0]):
        ax.plot(lines.z[line_index], lines.x[line_index], lines.y[line_index], color="tab:red", linewidth=1.0)
    ax.set_xlabel("z")
    ax.set_ylabel("x")
    ax.set_zlabel("y")
    ax.set_title("boundary B direction and field lines")
    ax.set_box_aspect([max(1.0, float(np.ptp(output.z))), 1, 1])
    ax.view_init(elev=18, azim=-62)
    fig.tight_layout()
    path = outdir / f"{_plot_name(output, name)}_mirror_bfield_boundary.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path
