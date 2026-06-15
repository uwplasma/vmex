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
