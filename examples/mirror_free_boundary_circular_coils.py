"""Sample circular-coil fields for the mirror free-boundary planning lane."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vmec_jax.mirror import (
    IPrimeProfile,
    MirrorCircularCoils,
    MirrorConfig,
    MirrorResolution,
    MirrorSolveOptions,
    PressureProfile,
    PsiPrimeProfile,
    initial_mirror_boundary_from_circular_coil_scan,
    load_mirror_output,
    make_mirror_free_boundary_circular_coil_scan,
    make_mirror_grid,
    mirror_external_bnormal,
    mirror_external_pressure_balance_response,
    mirror_lcfs_diagnostic,
    mirror_lcfs_merit,
    plot_mirror_output,
    propose_axisymmetric_mirror_lcfs_candidate_set,
    run_mirror_fixed_boundary,
    sample_mirror_axis_external_field,
    sample_mirror_boundary_external_field,
    two_coil_on_axis_bz,
    write_mirror_output,
    write_mirror_free_boundary_circular_coil_scan,
)


@dataclass(frozen=True)
class _LCFSProposalSelection:
    proposal: object
    candidate_summaries: list[dict[str, object]]
    allowed_strategies: tuple[str, ...]
    rejection_reason: str | None


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
    parser.add_argument("--run-fixed-boundary-baseline", action="store_true")
    parser.add_argument("--baseline-maxiter", type=int, default=0)
    parser.add_argument("--baseline-psi-prime", type=float, default=0.01)
    parser.add_argument("--lcfs-update-damping", type=float, default=0.25)
    parser.add_argument("--lcfs-update-max-relative-step", type=float, default=0.05)
    parser.add_argument("--lcfs-update-cap-taper-power", type=float, default=2.0)
    parser.add_argument("--lcfs-update-smoothing-passes", type=int, default=1)
    parser.add_argument("--lcfs-merit-bnormal-weight", type=float, default=1.0)
    parser.add_argument(
        "--lcfs-proposal-mode",
        choices=("best_predicted", "local", "scale", "bnormal", "mixed"),
        default="best_predicted",
    )
    parser.add_argument("--lcfs-require-bnormal-nonincrease", action="store_true")
    parser.add_argument("--run-lcfs-pilot", action="store_true")
    parser.add_argument("--lcfs-pilot-steps", type=int, default=1)
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


def _write_lcfs_diagnostic_plot(diagnostic, proposal=None, *, outdir: Path, name: str) -> Path:
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    z = np.asarray(diagnostic.z)
    theta = np.asarray(diagnostic.theta)
    bnormal = np.asarray(diagnostic.external_bnormal)
    pressure_balance = np.asarray(diagnostic.pressure_balance)
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 5.2), sharex=True)
    if theta.size == 1:
        axes[0].plot(z, bnormal[0], "o-", markersize=3.0)
        axes[1].plot(z, pressure_balance[0], "o-", markersize=3.0, label="before")
        if proposal is not None:
            axes[1].plot(
                np.asarray(proposal.z),
                np.asarray(proposal.pressure_balance_predicted),
                "s--",
                markersize=3.0,
                label="predicted update",
            )
    else:
        mesh0 = axes[0].pcolormesh(z, theta, bnormal, shading="auto")
        fig.colorbar(mesh0, ax=axes[0], label="B_ext . n")
        mesh1 = axes[1].pcolormesh(z, theta, pressure_balance, shading="auto")
        fig.colorbar(mesh1, ax=axes[1], label="pressure balance")
    axes[0].axhline(0.0, color="k", linewidth=0.8, alpha=0.6)
    axes[1].axhline(0.0, color="k", linewidth=0.8, alpha=0.6)
    axes[0].set_ylabel("B_ext . n")
    axes[1].set_ylabel("pressure balance")
    axes[1].set_xlabel("z")
    axes[0].set_title("LCFS target diagnostic")
    if proposal is not None and theta.size == 1:
        axes[1].legend(fontsize="small")
    fig.tight_layout()
    path = outdir / f"{name}_lcfs_diagnostic.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _beta_label(beta_percent: float) -> str:
    return f"{float(beta_percent):g}".replace(".", "p")


def _proposal_predicted_metrics(proposal, *, grid, coils, baseline_merit) -> dict[str, object]:
    sample = sample_mirror_boundary_external_field(grid, proposal.boundary, coils)
    boundary_r = proposal.boundary.radius_on_grid_3d(grid)
    bnormal = mirror_external_bnormal(boundary_r, grid.z, sample)
    bnormal_rms = float(np.sqrt(np.mean(np.asarray(bnormal, dtype=float) ** 2)))
    pressure_rms = float(proposal.pressure_balance_rms_predicted)
    pressure_term = pressure_rms / max(float(baseline_merit.pressure_scale), 1.0e-300)
    bnormal_term = bnormal_rms / max(float(baseline_merit.bnormal_scale), 1.0e-300)
    merit = float(np.sqrt(pressure_term**2 + float(baseline_merit.bnormal_weight) * bnormal_term**2))
    return {
        "strategy": str(proposal.strategy),
        "predicted_merit": merit,
        "predicted_pressure_balance_rms": pressure_rms,
        "predicted_external_bnormal_rms": bnormal_rms,
        "max_relative_delta_radius": float(
            np.max(np.abs(proposal.delta_radius) / np.maximum(proposal.old_radius, 1.0e-300))
        ),
    }


def _select_lcfs_proposal(
    *,
    lcfs,
    pressure_response,
    grid,
    coils,
    external_sample,
    baseline_merit,
    mode: str,
    damping: float,
    max_relative_step: float,
    cap_taper_power: float,
    smoothing_passes: int,
    require_bnormal_nonincrease: bool,
) -> _LCFSProposalSelection:
    candidates = list(
        propose_axisymmetric_mirror_lcfs_candidate_set(
            lcfs,
            external_sample,
            pressure_response,
            damping=damping,
            max_relative_step=max_relative_step,
            radius_floor=1.0e-4,
            preserve_caps=True,
            cap_taper_power=cap_taper_power,
            smoothing_passes=smoothing_passes,
            bnormal_weight=baseline_merit.bnormal_weight,
        )
    )
    summaries = [
        _proposal_predicted_metrics(candidate, grid=grid, coils=coils, baseline_merit=baseline_merit)
        for candidate in candidates
    ]
    mode_to_strategy = {
        "local": "local_pressure",
        "scale": "scale_pressure",
        "bnormal": "bnormal_slope",
        "mixed": "mixed_scale_bnormal",
    }
    if mode in mode_to_strategy:
        strategy = mode_to_strategy[mode]
        proposal = next(candidate for candidate in candidates if candidate.strategy == strategy)
        return _LCFSProposalSelection(
            proposal=proposal,
            candidate_summaries=summaries,
            allowed_strategies=tuple(summary["strategy"] for summary in summaries),
            rejection_reason=None,
        )
    allowed = np.ones(len(summaries), dtype=bool)
    if require_bnormal_nonincrease:
        baseline_bnormal = float(baseline_merit.external_bnormal_rms)
        allowed = np.asarray(
            [
                summary["strategy"] == "noop" or summary["predicted_external_bnormal_rms"] <= baseline_bnormal + 1.0e-14
                for summary in summaries
            ],
            dtype=bool,
        )
    merit_values = [summary["predicted_merit"] if allowed[index] else np.inf for index, summary in enumerate(summaries)]
    best_index = int(np.argmin(merit_values))
    allowed_strategies = tuple(
        str(summary["strategy"]) for index, summary in enumerate(summaries) if bool(allowed[index])
    )
    nonzero_allowed = any(strategy != "noop" for strategy in allowed_strategies)
    selected = candidates[best_index]
    rejection_reason = (
        "normal_field_guard_no_candidate"
        if require_bnormal_nonincrease and selected.strategy == "noop" and not nonzero_allowed
        else None
    )
    return _LCFSProposalSelection(
        proposal=selected,
        candidate_summaries=summaries,
        allowed_strategies=allowed_strategies,
        rejection_reason=rejection_reason,
    )


def _next_proposal_fields(selection: _LCFSProposalSelection) -> dict[str, object]:
    proposal = selection.proposal
    return {
        "lcfs_update_pressure_balance_rms_predicted_next": float(proposal.pressure_balance_rms_predicted),
        "lcfs_update_strategy_next": str(proposal.strategy),
        "lcfs_update_candidate_summaries_next": selection.candidate_summaries,
        "lcfs_update_allowed_strategies_next": list(selection.allowed_strategies),
        "lcfs_update_rejection_reason_next": selection.rejection_reason,
        "lcfs_update_cap_taper_power_next": float(proposal.cap_taper_power),
        "lcfs_update_smoothing_passes_next": int(proposal.smoothing_passes),
        "lcfs_update_max_relative_delta_radius_next": float(
            np.max(np.abs(proposal.delta_radius) / np.maximum(proposal.old_radius, 1.0e-300))
        ),
    }


def _skipped_lcfs_pilot_row(
    *,
    step: int,
    current_lcfs,
    current_merit,
    selection: _LCFSProposalSelection,
) -> dict[str, object]:
    row = {
        "step": int(step),
        "mout": None,
        "accepted": False,
        "skipped": True,
        "rejection_reason": selection.rejection_reason or "noop_candidate_selected",
        "final_residual_norm": None,
        "final_fsq": None,
        "final_normalized_force": None,
        "min_sqrtg": None,
        "mirror_ratio": None,
        "lcfs_external_bnormal_rms": float(current_lcfs.external_bnormal_rms),
        "lcfs_external_bnormal_max": float(current_lcfs.external_bnormal_max),
        "lcfs_pressure_balance_rms": float(current_lcfs.pressure_balance_rms),
        "lcfs_pressure_balance_max": float(current_lcfs.pressure_balance_max),
        "lcfs_pressure_balance_rms_change_fraction": 0.0,
        "lcfs_merit": float(current_merit.value),
        "lcfs_merit_change_fraction": 0.0,
        "lcfs_merit_bnormal_weight": float(current_merit.bnormal_weight),
        "figures": {},
    }
    row.update(_next_proposal_fields(selection))
    row["lcfs_update_max_relative_delta_radius_next"] = 0.0
    return row


def _completed_lcfs_pilot_row(
    *,
    step: int,
    mout,
    result,
    lcfs,
    merit,
    reference_lcfs,
    reference_merit,
    next_selection: _LCFSProposalSelection,
    accepted: bool,
    figures: dict[str, str],
) -> dict[str, object]:
    final = result.final_trace
    row = {
        "step": int(step),
        "mout": str(mout),
        "accepted": bool(accepted),
        "final_residual_norm": float(final.residual_norm),
        "final_fsq": float(final.fsq),
        "final_normalized_force": float(final.normalized_force),
        "min_sqrtg": float(final.min_sqrtg),
        "mirror_ratio": float(final.mirror_ratio),
        "lcfs_external_bnormal_rms": float(lcfs.external_bnormal_rms),
        "lcfs_external_bnormal_max": float(lcfs.external_bnormal_max),
        "lcfs_pressure_balance_rms": float(lcfs.pressure_balance_rms),
        "lcfs_pressure_balance_max": float(lcfs.pressure_balance_max),
        "lcfs_pressure_balance_rms_change_fraction": float(
            1.0 - lcfs.pressure_balance_rms / max(reference_lcfs.pressure_balance_rms, 1.0e-300)
        ),
        "lcfs_merit": float(merit.value),
        "lcfs_merit_change_fraction": float(1.0 - merit.value / max(reference_merit.value, 1.0e-300)),
        "lcfs_merit_bnormal_weight": float(merit.bnormal_weight),
        "figures": figures,
    }
    row.update(_next_proposal_fields(next_selection))
    return row


def _run_fixed_boundary_baseline_cases(
    *,
    outdir: Path,
    grid,
    boundary,
    scan,
    maxiter: int,
    psi_prime_value: float,
    lcfs_update_damping: float,
    lcfs_update_max_relative_step: float,
    lcfs_update_cap_taper_power: float,
    lcfs_update_smoothing_passes: int,
    lcfs_merit_bnormal_weight: float,
    lcfs_proposal_mode: str,
    lcfs_require_bnormal_nonincrease: bool,
    run_lcfs_pilot: bool,
    lcfs_pilot_steps: int,
    write_plots: bool,
) -> list[dict[str, object]]:
    config = MirrorConfig(
        MirrorResolution(ns=grid.ns, ntheta=1, nxi=grid.nxi, mpol=0),
        z_min=float(grid.z[0]),
        z_max=float(grid.z[-1]),
    )
    baseline_grid = make_mirror_grid(
        ns=grid.ns,
        ntheta=1,
        nxi=grid.nxi,
        mpol=0,
        z_min=float(grid.z[0]),
        z_max=float(grid.z[-1]),
    )
    external_sample = sample_mirror_boundary_external_field(baseline_grid, boundary, scan.coils)
    rows = []
    for case in scan.beta_cases:
        label = _beta_label(case.beta_percent)
        pressure = PressureProfile.polynomial([case.pressure_scale, -case.pressure_scale], gamma=2.0)
        solve_options = MirrorSolveOptions(optimizer="lbfgs", maxiter=int(maxiter), tolerance=1.0e-10, mu0=1.0)
        result = run_mirror_fixed_boundary(
            config,
            boundary,
            psi_prime=PsiPrimeProfile.constant(float(psi_prime_value)),
            i_prime=IPrimeProfile.zero(),
            pressure=pressure,
            options=solve_options,
        )
        mout_path = outdir / f"mout_free_boundary_circular_coils_beta_{label}.nc"
        if mout_path.exists():
            mout_path.unlink()
        mout = write_mirror_output(mout_path, result)
        output = load_mirror_output(mout)
        lcfs = mirror_lcfs_diagnostic(output, external_sample, mu0=1.0)
        lcfs_merit = mirror_lcfs_merit(lcfs, bnormal_weight=lcfs_merit_bnormal_weight)
        pressure_response = mirror_external_pressure_balance_response(lcfs, scan.coils, mu0=1.0)
        proposal_selection = _select_lcfs_proposal(
            lcfs=lcfs,
            pressure_response=pressure_response,
            grid=baseline_grid,
            coils=scan.coils,
            external_sample=external_sample,
            baseline_merit=lcfs_merit,
            mode=lcfs_proposal_mode,
            damping=lcfs_update_damping,
            max_relative_step=lcfs_update_max_relative_step,
            cap_taper_power=lcfs_update_cap_taper_power,
            smoothing_passes=lcfs_update_smoothing_passes,
            require_bnormal_nonincrease=lcfs_require_bnormal_nonincrease,
        )
        proposal = proposal_selection.proposal
        pilot_rows: list[dict[str, object]] = []
        accepted_merit_value = float(lcfs_merit.value)
        current_lcfs = lcfs
        current_merit = lcfs_merit
        candidate_proposal = proposal
        candidate_boundary = proposal.boundary
        candidate_selection = proposal_selection
        if run_lcfs_pilot:
            for step in range(1, int(lcfs_pilot_steps) + 1):
                if candidate_proposal.strategy == "noop":
                    pilot_rows.append(
                        _skipped_lcfs_pilot_row(
                            step=step,
                            current_lcfs=current_lcfs,
                            current_merit=current_merit,
                            selection=candidate_selection,
                        )
                    )
                    break
                pilot_result = run_mirror_fixed_boundary(
                    config,
                    candidate_boundary,
                    psi_prime=PsiPrimeProfile.constant(float(psi_prime_value)),
                    i_prime=IPrimeProfile.zero(),
                    pressure=pressure,
                    options=solve_options,
                )
                pilot_mout_path = outdir / f"mout_free_boundary_circular_coils_beta_{label}_lcfs_step_{step}.nc"
                if pilot_mout_path.exists():
                    pilot_mout_path.unlink()
                pilot_mout = write_mirror_output(pilot_mout_path, pilot_result)
                pilot_output = load_mirror_output(pilot_mout)
                pilot_external_sample = sample_mirror_boundary_external_field(
                    baseline_grid, candidate_boundary, scan.coils
                )
                pilot_lcfs = mirror_lcfs_diagnostic(pilot_output, pilot_external_sample, mu0=1.0)
                pilot_merit = mirror_lcfs_merit(
                    pilot_lcfs,
                    pressure_scale=lcfs_merit.pressure_scale,
                    bnormal_scale=lcfs_merit.bnormal_scale,
                    bnormal_weight=lcfs_merit_bnormal_weight,
                )
                pilot_response = mirror_external_pressure_balance_response(pilot_lcfs, scan.coils, mu0=1.0)
                pilot_selection = _select_lcfs_proposal(
                    lcfs=pilot_lcfs,
                    pressure_response=pilot_response,
                    grid=baseline_grid,
                    coils=scan.coils,
                    external_sample=pilot_external_sample,
                    baseline_merit=lcfs_merit,
                    mode=lcfs_proposal_mode,
                    damping=lcfs_update_damping,
                    max_relative_step=lcfs_update_max_relative_step,
                    cap_taper_power=lcfs_update_cap_taper_power,
                    smoothing_passes=lcfs_update_smoothing_passes,
                    require_bnormal_nonincrease=lcfs_require_bnormal_nonincrease,
                )
                pilot_proposal = pilot_selection.proposal
                pilot_plot_paths: dict[str, str] = {}
                if write_plots:
                    pilot_figure_dir = outdir / "figures" / f"fixed_boundary_beta_{label}_lcfs_step_{step}"
                    pilot_plot_paths = {
                        name: str(path)
                        for name, path in plot_mirror_output(pilot_mout, outdir=pilot_figure_dir).items()
                    }
                    pilot_plot_paths["lcfs_diagnostic"] = str(
                        _write_lcfs_diagnostic_plot(
                            pilot_lcfs,
                            pilot_proposal,
                            outdir=pilot_figure_dir,
                            name=f"free_boundary_circular_coils_beta_{label}_lcfs_step_{step}",
                        )
                    )
                pilot_final = pilot_result.final_trace
                accepted = bool(pilot_merit.value <= accepted_merit_value)
                pilot_rows.append(
                    _completed_lcfs_pilot_row(
                        step=step,
                        mout=pilot_mout,
                        result=pilot_result,
                        lcfs=pilot_lcfs,
                        merit=pilot_merit,
                        reference_lcfs=lcfs,
                        reference_merit=lcfs_merit,
                        next_selection=pilot_selection,
                        accepted=accepted,
                        figures=pilot_plot_paths,
                    )
                )
                if not accepted:
                    break
                accepted_merit_value = float(pilot_merit.value)
                current_lcfs = pilot_lcfs
                current_merit = pilot_merit
                candidate_proposal = pilot_proposal
                candidate_boundary = pilot_proposal.boundary
                candidate_selection = pilot_selection
        plot_paths: dict[str, str] = {}
        if write_plots:
            figure_dir = outdir / "figures" / f"fixed_boundary_beta_{label}"
            plot_paths = {name: str(path) for name, path in plot_mirror_output(mout, outdir=figure_dir).items()}
            plot_paths["lcfs_diagnostic"] = str(
                _write_lcfs_diagnostic_plot(
                    lcfs,
                    proposal,
                    outdir=figure_dir,
                    name=f"free_boundary_circular_coils_beta_{label}",
                )
            )
        summary = result.optimizer_summaries[-1] if result.optimizer_summaries else None
        final = result.final_trace
        rows.append(
            {
                "beta_percent": float(case.beta_percent),
                "beta_fraction": float(case.beta_fraction),
                "pressure_scale": float(case.pressure_scale),
                "mout": str(mout),
                "optimizer": str(summary.optimizer if summary is not None else "lbfgs"),
                "optimizer_success": bool(summary.success) if summary is not None else False,
                "optimizer_nit": int(summary.nit) if summary is not None else 0,
                "final_residual_norm": float(final.residual_norm),
                "final_fsq": float(final.fsq),
                "final_normalized_force": float(final.normalized_force),
                "min_sqrtg": float(final.min_sqrtg),
                "mirror_ratio": float(final.mirror_ratio),
                "lcfs_external_bnormal_rms": float(lcfs.external_bnormal_rms),
                "lcfs_external_bnormal_max": float(lcfs.external_bnormal_max),
                "lcfs_pressure_balance_rms": float(lcfs.pressure_balance_rms),
                "lcfs_pressure_balance_max": float(lcfs.pressure_balance_max),
                "lcfs_edge_pressure": float(lcfs.edge_pressure),
                "lcfs_merit": float(lcfs_merit.value),
                "lcfs_merit_pressure_scale": float(lcfs_merit.pressure_scale),
                "lcfs_merit_bnormal_scale": float(lcfs_merit.bnormal_scale),
                "lcfs_merit_bnormal_weight": float(lcfs_merit.bnormal_weight),
                "lcfs_pressure_response_min": float(np.min(proposal.pressure_response)),
                "lcfs_pressure_response_max": float(np.max(proposal.pressure_response)),
                "lcfs_update_pressure_balance_rms_predicted": float(proposal.pressure_balance_rms_predicted),
                "lcfs_update_strategy": str(proposal.strategy),
                "lcfs_update_candidate_summaries": proposal_selection.candidate_summaries,
                "lcfs_update_allowed_strategies": list(proposal_selection.allowed_strategies),
                "lcfs_update_rejection_reason": proposal_selection.rejection_reason,
                "lcfs_update_normal_field_guard": bool(lcfs_require_bnormal_nonincrease),
                "lcfs_update_pressure_balance_rms_reduction_fraction": float(
                    1.0 - proposal.pressure_balance_rms_predicted / max(proposal.pressure_balance_rms_before, 1.0e-300)
                ),
                "lcfs_update_cap_taper_power": float(proposal.cap_taper_power),
                "lcfs_update_smoothing_passes": int(proposal.smoothing_passes),
                "lcfs_update_max_abs_delta_radius": float(np.max(np.abs(proposal.delta_radius))),
                "lcfs_update_max_relative_delta_radius": float(
                    np.max(np.abs(proposal.delta_radius) / np.maximum(proposal.old_radius, 1.0e-300))
                ),
                "lcfs_pilot_rows": pilot_rows,
                "figures": plot_paths,
            }
        )
    return rows


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
    run_fixed_boundary_baseline: bool = False,
    baseline_maxiter: int = 0,
    baseline_psi_prime: float = 0.01,
    lcfs_update_damping: float = 0.25,
    lcfs_update_max_relative_step: float = 0.05,
    lcfs_update_cap_taper_power: float = 2.0,
    lcfs_update_smoothing_passes: int = 1,
    lcfs_merit_bnormal_weight: float = 1.0,
    lcfs_proposal_mode: str = "best_predicted",
    lcfs_require_bnormal_nonincrease: bool = False,
    run_lcfs_pilot: bool = False,
    lcfs_pilot_steps: int = 1,
    write_plots: bool = True,
) -> Path:
    if run_lcfs_pilot and int(lcfs_pilot_steps) < 1:
        raise ValueError("lcfs_pilot_steps must be at least 1 when run_lcfs_pilot is enabled")
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
    baseline_rows = (
        _run_fixed_boundary_baseline_cases(
            outdir=outdir,
            grid=grid,
            boundary=boundary,
            scan=scan,
            maxiter=baseline_maxiter,
            psi_prime_value=baseline_psi_prime,
            lcfs_update_damping=lcfs_update_damping,
            lcfs_update_max_relative_step=lcfs_update_max_relative_step,
            lcfs_update_cap_taper_power=lcfs_update_cap_taper_power,
            lcfs_update_smoothing_passes=lcfs_update_smoothing_passes,
            lcfs_merit_bnormal_weight=lcfs_merit_bnormal_weight,
            lcfs_proposal_mode=lcfs_proposal_mode,
            lcfs_require_bnormal_nonincrease=lcfs_require_bnormal_nonincrease,
            run_lcfs_pilot=run_lcfs_pilot,
            lcfs_pilot_steps=lcfs_pilot_steps,
            write_plots=write_plots,
        )
        if run_fixed_boundary_baseline
        else []
    )

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
        "fixed_boundary_baseline_rows": baseline_rows,
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
        run_fixed_boundary_baseline=args.run_fixed_boundary_baseline,
        baseline_maxiter=args.baseline_maxiter,
        baseline_psi_prime=args.baseline_psi_prime,
        lcfs_update_damping=args.lcfs_update_damping,
        lcfs_update_max_relative_step=args.lcfs_update_max_relative_step,
        lcfs_update_cap_taper_power=args.lcfs_update_cap_taper_power,
        lcfs_update_smoothing_passes=args.lcfs_update_smoothing_passes,
        lcfs_merit_bnormal_weight=args.lcfs_merit_bnormal_weight,
        lcfs_proposal_mode=args.lcfs_proposal_mode,
        lcfs_require_bnormal_nonincrease=args.lcfs_require_bnormal_nonincrease,
        run_lcfs_pilot=args.run_lcfs_pilot,
        lcfs_pilot_steps=args.lcfs_pilot_steps,
        write_plots=not args.no_plots,
    )
    print(path)


if __name__ == "__main__":
    main()
