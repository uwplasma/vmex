"""Diagnostic plot-data helpers for mirror output files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..io.mout import load_mirror_output
from ..io.schema import MirrorOutput
from .geometry import _import_matplotlib, _plot_name


@dataclass(frozen=True)
class MirrorJacobianData:
    """Theta-averaged Jacobian data."""

    s: np.ndarray
    xi: np.ndarray
    sqrtg: np.ndarray
    min_sqrtg: float
    max_sqrtg: float


@dataclass(frozen=True)
class MirrorPressureProfileData:
    """Radial pressure and beta profile data."""

    s: np.ndarray
    pressure: np.ndarray
    dpressure_ds: np.ndarray
    beta: np.ndarray


@dataclass(frozen=True)
class MirrorResidualHistoryData:
    """Residual and energy solve-history data."""

    index: np.ndarray
    residual_norm: np.ndarray
    fsq: np.ndarray
    normalized_force: np.ndarray
    energy_total: np.ndarray
    step_size: np.ndarray
    pressure_scale: np.ndarray


@dataclass(frozen=True)
class MirrorRadialDiagnosticsData:
    """Radial beta, twist, and magnetic-well proxy diagnostics."""

    s: np.ndarray
    beta: np.ndarray
    iota_like_twist: np.ndarray
    mean_bmag: np.ndarray
    magnetic_well_proxy: np.ndarray


def _as_output(output_or_path) -> MirrorOutput:
    return output_or_path if isinstance(output_or_path, MirrorOutput) else load_mirror_output(output_or_path)


def mirror_jacobian_data(output_or_path) -> MirrorJacobianData:
    """Return theta-averaged Jacobian plot data."""
    output = _as_output(output_or_path)
    sqrtg = np.mean(np.asarray(output.geometry.sqrtg), axis=1)
    return MirrorJacobianData(
        s=np.asarray(output.s),
        xi=np.asarray(output.xi),
        sqrtg=sqrtg,
        min_sqrtg=float(np.min(sqrtg)),
        max_sqrtg=float(np.max(sqrtg)),
    )


def mirror_pressure_profile_data(output_or_path) -> MirrorPressureProfileData:
    """Return radial pressure-profile plot data."""
    output = _as_output(output_or_path)
    return MirrorPressureProfileData(
        s=np.asarray(output.s),
        pressure=np.asarray(output.profiles.pressure),
        dpressure_ds=np.asarray(output.profiles.dpressure_ds),
        beta=np.asarray(output.profiles.beta),
    )


def mirror_residual_history_data(output_or_path) -> MirrorResidualHistoryData:
    """Return residual/energy solve-history plot data."""
    output = _as_output(output_or_path)
    return MirrorResidualHistoryData(
        index=np.arange(output.history.residual_norm.size),
        residual_norm=np.asarray(output.history.residual_norm),
        fsq=np.asarray(output.history.fsq),
        normalized_force=np.asarray(output.history.normalized_force),
        energy_total=np.asarray(output.history.energy_total),
        step_size=np.asarray(output.history.step_size),
        pressure_scale=np.asarray(output.history.pressure_scale),
    )


def mirror_radial_diagnostics_data(output_or_path) -> MirrorRadialDiagnosticsData:
    """Return radial diagnostics for open-ended mirror outputs.

    The ``iota_like_twist`` field is the profile ratio ``I'/Psi'``.  It is a
    twist proxy, not toroidal rotational transform, because mirror field lines
    are open and the axial coordinate is nonperiodic.
    """
    output = _as_output(output_or_path)
    bmag = np.asarray(output.field.bmag)
    weights = output.w_theta[:, None] * output.w_xi[None, :]
    mean_bmag = np.einsum("jk,ijk->i", weights, bmag) / np.sum(weights)
    with np.errstate(divide="ignore", invalid="ignore"):
        twist = np.divide(
            output.profiles.i_prime,
            output.profiles.psi_prime,
            out=np.zeros_like(output.profiles.i_prime),
            where=np.abs(output.profiles.psi_prime) > 0.0,
        )
    magnetic_well_proxy = -np.gradient(mean_bmag, output.s, edge_order=1)
    return MirrorRadialDiagnosticsData(
        s=np.asarray(output.s),
        beta=np.asarray(output.profiles.beta),
        iota_like_twist=twist,
        mean_bmag=mean_bmag,
        magnetic_well_proxy=magnetic_well_proxy,
    )


def write_mirror_jacobian(output_or_path, *, outdir: str | Path, name: str | None = None) -> Path:
    """Write the theta-averaged Jacobian map."""
    output = _as_output(output_or_path)
    data = mirror_jacobian_data(output)
    plt = _import_matplotlib()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    mesh = ax.pcolormesh(data.xi, data.s, data.sqrtg, shading="auto")
    ax.set_xlabel("xi")
    ax.set_ylabel("s")
    ax.set_title(f"sqrt(g) [{data.min_sqrtg:.3g}, {data.max_sqrtg:.3g}]")
    fig.colorbar(mesh, ax=ax, label="sqrt(g)")
    fig.tight_layout()
    path = outdir / f"{_plot_name(output, name)}_mirror_jacobian.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def write_mirror_pressure_profile(output_or_path, *, outdir: str | Path, name: str | None = None) -> Path:
    """Write radial pressure and beta profiles."""
    output = _as_output(output_or_path)
    data = mirror_pressure_profile_data(output)
    plt = _import_matplotlib()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(data.s, data.pressure, ".-", label="p")
    ax.plot(data.s, data.dpressure_ds, ".-", label="dp/ds")
    ax.plot(data.s, data.beta, ".-", label="beta")
    ax.set_xlabel("s")
    ax.legend(fontsize="x-small")
    fig.tight_layout()
    path = outdir / f"{_plot_name(output, name)}_mirror_pressure_profile.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def write_mirror_residual_history(output_or_path, *, outdir: str | Path, name: str | None = None) -> Path:
    """Write residual and energy history diagnostics."""
    output = _as_output(output_or_path)
    data = mirror_residual_history_data(output)
    plt = _import_matplotlib()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(6, 4.5), sharex=True)
    ax = axes[0]
    ax.semilogy(data.index, np.maximum(data.residual_norm, 1.0e-300), ".-", label="residual")
    ax.semilogy(data.index, np.maximum(data.normalized_force, 1.0e-300), ".-", label="normalized force")
    ax.semilogy(data.index, np.maximum(data.fsq, 1.0e-300), ".-", label="mirror fsq")
    positive_step = np.where(data.step_size > 0.0, data.step_size, np.nan)
    ax.semilogy(data.index, positive_step, ".-", label="step norm")
    ax.set_ylabel("norm")
    ax.set_title("fixed-boundary convergence")
    ax.legend(fontsize="x-small")

    ax_energy = axes[1]
    ax_energy.plot(data.index, data.energy_total, ".-", color="tab:orange")
    ax_energy.set_xlabel("history index")
    ax_energy.set_ylabel("total energy")
    ax_energy.ticklabel_format(axis="y", useOffset=False)
    fig.tight_layout()
    path = outdir / f"{_plot_name(output, name)}_mirror_residual_history.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def write_mirror_radial_diagnostics(output_or_path, *, outdir: str | Path, name: str | None = None) -> Path:
    """Write radial beta, twist-proxy, and magnetic-well proxy diagnostics."""
    output = _as_output(output_or_path)
    data = mirror_radial_diagnostics_data(output)
    plt = _import_matplotlib()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(7, 5), sharex=True)
    axes[0, 0].plot(data.s, data.beta, ".-")
    axes[0, 0].set_ylabel("beta")
    axes[0, 1].plot(data.s, data.iota_like_twist, ".-")
    axes[0, 1].set_ylabel("I'/Psi'")
    axes[0, 1].set_title("open-field twist proxy")
    axes[1, 0].plot(data.s, data.mean_bmag, ".-")
    axes[1, 0].set_ylabel("<|B|>")
    axes[1, 1].plot(data.s, data.magnetic_well_proxy, ".-")
    axes[1, 1].set_ylabel("-d<|B|>/ds")
    axes[1, 1].set_title("magnetic-well proxy")
    for ax in axes[-1, :]:
        ax.set_xlabel("s")
    fig.tight_layout()
    path = outdir / f"{_plot_name(output, name)}_mirror_radial_diagnostics.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path
