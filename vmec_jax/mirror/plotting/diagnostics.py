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
    energy_total: np.ndarray
    pressure_scale: np.ndarray


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
        energy_total=np.asarray(output.history.energy_total),
        pressure_scale=np.asarray(output.history.pressure_scale),
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

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.semilogy(data.index, np.maximum(data.residual_norm, 1.0e-300), ".-", label="residual")
    ax.set_xlabel("history index")
    ax.set_ylabel("residual norm")
    ax2 = ax.twinx()
    ax2.plot(data.index, data.energy_total, ".-", color="tab:orange", label="energy")
    ax2.set_ylabel("total energy")
    fig.tight_layout()
    path = outdir / f"{_plot_name(output, name)}_mirror_residual_history.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path
