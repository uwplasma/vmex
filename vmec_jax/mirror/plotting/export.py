"""High-level plotting and export helpers for mirror output files."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..io.mout import load_mirror_output
from ..io.schema import MirrorOutput
from .bfield import write_mirror_bmag_boundary, write_mirror_bmag_sxi
from .diagnostics import write_mirror_jacobian, write_mirror_pressure_profile, write_mirror_residual_history
from .geometry import write_mirror_boundary_3d, write_mirror_surfaces_rz


def _as_output(output_or_path) -> MirrorOutput:
    return output_or_path if isinstance(output_or_path, MirrorOutput) else load_mirror_output(output_or_path)


def _plot_name(output: MirrorOutput, name: str | None) -> str:
    if name is not None:
        return str(name)
    if output.path is None:
        return "mirror"
    stem = output.path.stem
    return stem[5:] if stem.startswith("mout_") else stem


def plot_mirror_output(
    output_or_path,
    *,
    outdir: str | Path | None = None,
    name: str | None = None,
    show: bool = False,
) -> dict[str, Path]:
    """Write the standard diagnostic plots for a mirror ``mout`` file."""
    output = _as_output(output_or_path)
    if outdir is None:
        outdir = output.path.parent if output.path is not None else Path.cwd()
    outdir = Path(outdir)
    plot_name = _plot_name(output, name)
    paths = {
        "surfaces_rz": write_mirror_surfaces_rz(output, outdir=outdir, name=plot_name),
        "boundary_3d": write_mirror_boundary_3d(output, outdir=outdir, name=plot_name),
        "bmag_sxi": write_mirror_bmag_sxi(output, outdir=outdir, name=plot_name),
        "bmag_boundary": write_mirror_bmag_boundary(output, outdir=outdir, name=plot_name),
        "jacobian": write_mirror_jacobian(output, outdir=outdir, name=plot_name),
        "pressure_profile": write_mirror_pressure_profile(output, outdir=outdir, name=plot_name),
        "residual_history": write_mirror_residual_history(output, outdir=outdir, name=plot_name),
    }
    if show:
        import matplotlib.pyplot as plt

        plt.show()
    return paths


def mirror_output_to_npz(output_or_path, path: str | Path) -> Path:
    """Export core mirror output arrays to ``.npz`` for lightweight inspection."""
    output = _as_output(output_or_path)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        s=output.s,
        theta=output.theta,
        xi=output.xi,
        z=output.z,
        r=output.geometry.r,
        sqrtg=output.geometry.sqrtg,
        bmag=output.field.bmag,
        pressure=output.profiles.pressure,
        residual_norm=output.history.residual_norm,
        energy_total=output.history.energy_total,
    )
    return path


def mirror_axisym_slice_to_csv(output_or_path, path: str | Path) -> Path:
    """Export the theta-zero axisymmetric slice to CSV columns ``s,xi,z,r,Bmag,sqrtg``."""
    output = _as_output(output_or_path)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    s2, xi2 = np.meshgrid(output.s, output.xi, indexing="ij")
    z2 = np.broadcast_to(output.z[None, :], s2.shape)
    table = np.column_stack(
        [
            s2.ravel(),
            xi2.ravel(),
            z2.ravel(),
            output.geometry.r[:, 0, :].ravel(),
            output.field.bmag[:, 0, :].ravel(),
            output.geometry.sqrtg[:, 0, :].ravel(),
        ]
    )
    np.savetxt(path, table, delimiter=",", header="s,xi,z,r,Bmag,sqrtg", comments="")
    return path
