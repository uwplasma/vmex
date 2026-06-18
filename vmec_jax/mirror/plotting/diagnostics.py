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
    field_line_theta_advance: np.ndarray
    field_line_turns: np.ndarray
    mean_bmag: np.ndarray
    magnetic_well_proxy: np.ndarray


@dataclass(frozen=True)
class MirrorBoozerLikeDiagnosticsData:
    """Flux-surface averages and pitch proxies for open mirror fields.

    These are not Boozer coordinates. They are mirror-native profile diagnostics
    that use the same style of quantities, such as flux-surface averaged
    ``|B|`` and field-line pitch, without assuming toroidal closure.
    """

    s: np.ndarray
    surface_measure: np.ndarray
    bmag_flux_surface_average: np.ndarray
    bmag_min: np.ndarray
    bmag_max: np.ndarray
    surface_mirror_ratio: np.ndarray
    normalized_bmag_ripple_rms: np.ndarray
    iota_like_twist: np.ndarray
    field_line_turns: np.ndarray
    contravariant_pitch_mean: np.ndarray
    contravariant_pitch_rms: np.ndarray
    covariant_pitch_ratio: np.ndarray
    magnetic_well_proxy: np.ndarray


@dataclass(frozen=True)
class MirrorFieldLinePitchProfileData:
    """Radial cap-to-cap field-line pitch profile."""

    s: np.ndarray
    theta_advance_mean: np.ndarray
    theta_advance_min: np.ndarray
    theta_advance_max: np.ndarray
    turns_mean: np.ndarray


def _as_output(output_or_path) -> MirrorOutput:
    return output_or_path if isinstance(output_or_path, MirrorOutput) else load_mirror_output(output_or_path)


def _interp_periodic(theta_nodes, values, theta_value: float) -> float:
    theta_nodes = np.asarray(theta_nodes, dtype=float)
    values = np.asarray(values, dtype=float)
    period = 2.0 * np.pi
    theta_wrapped = float(np.mod(theta_value, period))
    extended_theta = np.concatenate([theta_nodes, theta_nodes[:1] + period])
    extended_values = np.concatenate([values, values[:1]])
    return float(np.interp(theta_wrapped, extended_theta, extended_values))


def _radial_gradient(values: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Return a radial gradient, including the single-surface edge case."""
    values = np.asarray(values, dtype=float)
    s = np.asarray(s, dtype=float)
    if values.size < 2:
        return np.zeros_like(values)
    return np.gradient(values, s, edge_order=1)


def _surface_quadrature_weights(output: MirrorOutput) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized Jacobian-weighted surface weights and measures."""
    theta_xi_weights = np.asarray(output.w_theta, dtype=float)[:, None] * np.asarray(output.w_xi, dtype=float)[None, :]
    raw_weights = np.asarray(output.geometry.sqrtg, dtype=float) * theta_xi_weights[None, :, :]
    surface_measure = np.sum(raw_weights, axis=(1, 2))
    weights = np.divide(
        raw_weights,
        surface_measure[:, None, None],
        out=np.zeros_like(raw_weights),
        where=np.abs(surface_measure[:, None, None]) > np.finfo(float).tiny,
    )
    return weights, surface_measure


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


def mirror_field_line_pitch_profile_data(output_or_path, *, num_lines: int = 6) -> MirrorFieldLinePitchProfileData:
    """Return cap-to-cap field-line pitch on each radial surface.

    This is an open-field-line diagnostic: it measures how much a traced line
    advances in poloidal angle between the two mirror caps.  It is not a
    toroidal rotational transform.
    """
    output = _as_output(output_or_path)
    num_lines = max(1, int(num_lines))
    theta_nodes = np.asarray(output.theta, dtype=float)
    xi = np.asarray(output.xi, dtype=float)
    start_theta = np.linspace(0.0, 2.0 * np.pi, num_lines, endpoint=False)
    theta_advance = np.zeros((output.ns, num_lines), dtype=float)
    btheta_all = np.asarray(output.field.b_sup_theta, dtype=float)
    bxi_all = np.asarray(output.field.b_sup_xi, dtype=float)

    for surface_index in range(output.ns):
        theta_lines = np.zeros((num_lines, output.nxi), dtype=float)
        theta_lines[:, 0] = start_theta
        btheta = btheta_all[surface_index]
        bxi = bxi_all[surface_index]
        for line_index in range(num_lines):
            for k in range(output.nxi - 1):
                numerator = _interp_periodic(theta_nodes, btheta[:, k], theta_lines[line_index, k])
                denominator = _interp_periodic(theta_nodes, bxi[:, k], theta_lines[line_index, k])
                slope = 0.0 if abs(denominator) <= np.finfo(float).tiny else numerator / denominator
                theta_lines[line_index, k + 1] = theta_lines[line_index, k] + slope * (xi[k + 1] - xi[k])
        theta_advance[surface_index] = theta_lines[:, -1] - theta_lines[:, 0]

    return MirrorFieldLinePitchProfileData(
        s=np.asarray(output.s),
        theta_advance_mean=np.mean(theta_advance, axis=1),
        theta_advance_min=np.min(theta_advance, axis=1),
        theta_advance_max=np.max(theta_advance, axis=1),
        turns_mean=np.mean(theta_advance, axis=1) / (2.0 * np.pi),
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
    magnetic_well_proxy = -_radial_gradient(mean_bmag, output.s)
    pitch = mirror_field_line_pitch_profile_data(output)
    return MirrorRadialDiagnosticsData(
        s=np.asarray(output.s),
        beta=np.asarray(output.profiles.beta),
        iota_like_twist=twist,
        field_line_theta_advance=pitch.theta_advance_mean,
        field_line_turns=pitch.turns_mean,
        mean_bmag=mean_bmag,
        magnetic_well_proxy=magnetic_well_proxy,
    )


def mirror_boozer_like_diagnostics_data(output_or_path) -> MirrorBoozerLikeDiagnosticsData:
    """Return mirror-native flux-surface diagnostics inspired by Boozer plots.

    Open mirrors do not have toroidal Boozer coordinates or rotational
    transform. This helper therefore reports Jacobian-weighted surface averages,
    cap-to-cap field-line turns, and pitch-variation proxies that are useful for
    comparing mirror equilibria without changing their geometry model.
    """
    output = _as_output(output_or_path)
    weights, surface_measure = _surface_quadrature_weights(output)
    bmag = np.asarray(output.field.bmag, dtype=float)
    bmag_average = np.sum(weights * bmag, axis=(1, 2))
    bmag_min = np.min(bmag, axis=(1, 2))
    bmag_max = np.max(bmag, axis=(1, 2))
    surface_mirror_ratio = np.divide(
        bmag_max,
        bmag_min,
        out=np.full_like(bmag_max, np.inf),
        where=np.abs(bmag_min) > np.finfo(float).tiny,
    )
    ripple = np.sqrt(np.sum(weights * (bmag - bmag_average[:, None, None]) ** 2, axis=(1, 2)))
    normalized_ripple = np.divide(
        ripple,
        bmag_average,
        out=np.zeros_like(ripple),
        where=np.abs(bmag_average) > np.finfo(float).tiny,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        iota_like_twist = np.divide(
            output.profiles.i_prime,
            output.profiles.psi_prime,
            out=np.zeros_like(output.profiles.i_prime),
            where=np.abs(output.profiles.psi_prime) > 0.0,
        )
        local_pitch = np.divide(
            output.field.b_sup_theta,
            output.field.b_sup_xi,
            out=np.zeros_like(output.field.b_sup_theta),
            where=np.abs(output.field.b_sup_xi) > np.finfo(float).tiny,
        )
    pitch_mean = np.sum(weights * local_pitch, axis=(1, 2))
    pitch_rms = np.sqrt(np.sum(weights * (local_pitch - pitch_mean[:, None, None]) ** 2, axis=(1, 2)))
    b_cov_theta_average = np.sum(weights * output.field.b_cov_theta, axis=(1, 2))
    b_cov_xi_average = np.sum(weights * output.field.b_cov_xi, axis=(1, 2))
    covariant_pitch_ratio = np.divide(
        b_cov_theta_average,
        b_cov_xi_average,
        out=np.zeros_like(b_cov_theta_average),
        where=np.abs(b_cov_xi_average) > np.finfo(float).tiny,
    )
    field_line_turns = mirror_field_line_pitch_profile_data(output).turns_mean
    return MirrorBoozerLikeDiagnosticsData(
        s=np.asarray(output.s),
        surface_measure=surface_measure,
        bmag_flux_surface_average=bmag_average,
        bmag_min=bmag_min,
        bmag_max=bmag_max,
        surface_mirror_ratio=surface_mirror_ratio,
        normalized_bmag_ripple_rms=normalized_ripple,
        iota_like_twist=iota_like_twist,
        field_line_turns=field_line_turns,
        contravariant_pitch_mean=pitch_mean,
        contravariant_pitch_rms=pitch_rms,
        covariant_pitch_ratio=covariant_pitch_ratio,
        magnetic_well_proxy=-_radial_gradient(bmag_average, output.s),
    )


def mirror_boozer_like_summary_metrics(output_or_path) -> dict[str, float]:
    """Return scalar extrema for the mirror-Boozer-like profile diagnostics."""
    data = mirror_boozer_like_diagnostics_data(output_or_path)
    return {
        "boozer_like_bmag_average_min": float(np.min(data.bmag_flux_surface_average)),
        "boozer_like_bmag_average_max": float(np.max(data.bmag_flux_surface_average)),
        "boozer_like_bmag_min_global": float(np.min(data.bmag_min)),
        "boozer_like_bmag_max_global": float(np.max(data.bmag_max)),
        "boozer_like_surface_mirror_ratio_min": float(np.min(data.surface_mirror_ratio)),
        "boozer_like_surface_mirror_ratio_max": float(np.max(data.surface_mirror_ratio)),
        "boozer_like_bmag_ripple_rms_max": float(np.max(data.normalized_bmag_ripple_rms)),
        "boozer_like_iota_like_twist_mean": float(np.mean(data.iota_like_twist)),
        "boozer_like_iota_like_twist_min": float(np.min(data.iota_like_twist)),
        "boozer_like_iota_like_twist_max": float(np.max(data.iota_like_twist)),
        "boozer_like_field_line_turns_mean": float(np.mean(data.field_line_turns)),
        "boozer_like_field_line_turns_min": float(np.min(data.field_line_turns)),
        "boozer_like_field_line_turns_max": float(np.max(data.field_line_turns)),
        "boozer_like_contravariant_pitch_mean_min": float(np.min(data.contravariant_pitch_mean)),
        "boozer_like_contravariant_pitch_mean_max": float(np.max(data.contravariant_pitch_mean)),
        "boozer_like_contravariant_pitch_rms_max": float(np.max(data.contravariant_pitch_rms)),
        "boozer_like_covariant_pitch_ratio_min": float(np.min(data.covariant_pitch_ratio)),
        "boozer_like_covariant_pitch_ratio_max": float(np.max(data.covariant_pitch_ratio)),
        "boozer_like_magnetic_well_proxy_min": float(np.min(data.magnetic_well_proxy)),
        "boozer_like_magnetic_well_proxy_max": float(np.max(data.magnetic_well_proxy)),
    }


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
    axes[0, 1].plot(data.s, data.iota_like_twist, ".-", label="I'/Psi'")
    axes[0, 1].plot(data.s, data.field_line_turns, ".-", label="cap-to-cap turns")
    axes[0, 1].set_ylabel("twist")
    axes[0, 1].set_title("open-field pitch")
    axes[0, 1].legend(fontsize="x-small")
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


def write_mirror_boozer_like_diagnostics(output_or_path, *, outdir: str | Path, name: str | None = None) -> Path:
    """Write open-mirror flux-surface averages and pitch proxies."""
    output = _as_output(output_or_path)
    data = mirror_boozer_like_diagnostics_data(output)
    plt = _import_matplotlib()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(9, 5.5), sharex=True)
    axes[0, 0].plot(data.s, data.bmag_flux_surface_average, ".-", label="<|B|>")
    axes[0, 0].plot(data.s, data.bmag_min, ".--", label="min |B|")
    axes[0, 0].plot(data.s, data.bmag_max, ".--", label="max |B|")
    axes[0, 0].set_ylabel("|B|")
    axes[0, 0].legend(fontsize="x-small")

    axes[0, 1].plot(data.s, data.surface_mirror_ratio, ".-", label="mirror ratio")
    axes[0, 1].plot(data.s, data.normalized_bmag_ripple_rms, ".-", label="ripple RMS")
    axes[0, 1].set_ylabel("B variation")
    axes[0, 1].legend(fontsize="x-small")

    axes[0, 2].plot(data.s, data.iota_like_twist, ".-", label="I'/Psi'")
    axes[0, 2].plot(data.s, data.field_line_turns, ".-", label="cap turns")
    axes[0, 2].set_ylabel("twist")
    axes[0, 2].set_title("open-field pitch")
    axes[0, 2].legend(fontsize="x-small")

    axes[1, 0].plot(data.s, data.contravariant_pitch_mean, ".-", label="<B^theta/B^xi>")
    axes[1, 0].plot(data.s, data.contravariant_pitch_rms, ".-", label="pitch RMS")
    axes[1, 0].set_ylabel("contravariant pitch")
    axes[1, 0].legend(fontsize="x-small")

    axes[1, 1].plot(data.s, data.covariant_pitch_ratio, ".-")
    axes[1, 1].set_ylabel("<B_theta>/<B_xi>")
    axes[1, 1].set_title("covariant pitch proxy")

    axes[1, 2].plot(data.s, data.magnetic_well_proxy, ".-")
    axes[1, 2].set_ylabel("-d<|B|>/ds")
    axes[1, 2].set_title("well proxy")

    for ax in axes[-1, :]:
        ax.set_xlabel("s")
    fig.suptitle("mirror Boozer-like diagnostics")
    fig.tight_layout()
    path = outdir / f"{_plot_name(output, name)}_mirror_boozer_like_diagnostics.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path
