"""Run two-coil residual-Newton convergence grids for mirror fixed-boundary solves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
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
    MirrorStateAxisym,
    PressureProfile,
    PsiPrimeProfile,
    load_mirror_output,
    mirror_boundary_from_on_axis_bz,
    plot_mirror_output,
    run_mirror_fixed_boundary,
    two_coil_on_axis_bz,
    write_mirror_output,
)
from vmec_jax.mirror.kernels.forces import axisym_projected_energy_residual


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("results/mirror/residual_newton_convergence_grid"))
    parser.add_argument("--ns-array", type=str, default="5,9")
    parser.add_argument("--nxi-array", type=str, default="9,17")
    parser.add_argument("--maxiter-array", type=str, default="6,12")
    parser.add_argument("--residual-linear-maxiter-array", type=str, default="16,48")
    parser.add_argument(
        "--residual-linear-maxiter-policy",
        type=str,
        default="fixed",
        choices=("fixed", "adaptive"),
    )
    parser.add_argument("--residual-linear-adaptive-factor", type=float, default=6.0)
    parser.add_argument(
        "--residual-linear-solver",
        type=str,
        default="lsmr",
        choices=("lsmr", "dense_lstsq"),
        help="Linear solver for each residual-Newton correction; dense_lstsq is intended for small reference grids.",
    )
    parser.add_argument("--preconditioners", type=str, default="radial_xi_tridi")
    parser.add_argument("--residual-radial-alpha", type=float, default=0.5)
    parser.add_argument("--residual-lambda-alpha", type=float, default=0.5)
    parser.add_argument("--residual-xi-alpha", type=float, default=0.2)
    parser.add_argument("--line-search-steps", type=int, default=32)
    parser.add_argument("--gtol", type=float, default=1.0e-12)
    parser.add_argument("--ftol", type=float, default=1.0e-12)
    parser.add_argument("--coil-radius", type=float, default=0.35)
    parser.add_argument("--separation", type=float, default=2.0)
    parser.add_argument("--current", type=float, default=1.0e6)
    parser.add_argument("--midplane-radius", type=float, default=0.3)
    parser.add_argument("--i-prime", type=float, default=0.0)
    parser.add_argument(
        "--case-label",
        type=str,
        default="",
        help="Label used for selected output folders; defaults to two_coil or finite_current_two_coil.",
    )
    parser.add_argument("--perturbation", type=float, default=0.02)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def _parse_int_array(value: str, *, name: str, minimum: int) -> tuple[int, ...]:
    items = [item for item in re.split(r"[\s,]+", str(value).strip()) if item]
    if not items:
        raise ValueError(f"{name} must contain at least one integer")
    values = tuple(int(item) for item in items)
    if any(item < minimum for item in values):
        raise ValueError(f"all {name} values must be at least {minimum}")
    return values


def _parse_preconditioners(value: str) -> tuple[str, ...]:
    allowed = {"none", "radial_tridi", "radial_xi_tridi", "radial_xi_lambda_xi_tridi"}
    values = tuple(item.strip().lower().replace("-", "_") for item in str(value).split(",") if item.strip())
    if not values:
        raise ValueError("preconditioners must contain at least one value")
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unsupported preconditioner(s): {', '.join(unknown)}")
    return values


def _safe_label(value: str) -> str:
    label = re.sub(r"[^0-9A-Za-z_]+", "_", str(value).strip().replace("-", "_")).strip("_").lower()
    return label or "two_coil"


def _default_case_label(i_prime: float) -> str:
    return "finite_current_two_coil" if abs(float(i_prime)) > 0.0 else "two_coil"


def _perturbed_initial_state(config: MirrorConfig, boundary, *, amplitude: float) -> MirrorStateAxisym:
    grid = config.build_grid()
    base = MirrorStateAxisym.from_boundary(grid, boundary)
    s = grid.s_full[:, None]
    xi = grid.xi[None, :]
    shape = s * (1.0 - s) * (1.0 - xi**2)
    a = base.a * (1.0 + float(amplitude) * shape)
    return MirrorStateAxisym(a=a, lam=np.zeros_like(a))


def _two_coil_problem(
    *,
    ns: int,
    nxi: int,
    coil_radius: float,
    separation: float,
    current: float,
    midplane_radius: float,
    i_prime: float,
    perturbation: float,
):
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
    return {
        "config": config,
        "boundary": boundary,
        "initial_state": _perturbed_initial_state(config, boundary, amplitude=perturbation),
        "psi_prime": PsiPrimeProfile.constant(psi_value),
        "i_prime": IPrimeProfile.constant(i_prime) if abs(float(i_prime)) > 0.0 else IPrimeProfile.zero(),
        "pressure": PressureProfile.zero(),
        "psi_value": float(psi_value),
        "i_prime_value": float(i_prime),
        "twist_proxy_i_prime_over_psi_prime": float(i_prime) / max(float(psi_value), np.finfo(float).tiny),
    }


def _history_rows(result, *, row_id: str) -> list[dict[str, object]]:
    return [
        {
            "row_id": row_id,
            "iteration": int(row.iteration),
            "energy_total": float(row.energy_total),
            "residual_norm": float(row.residual_norm),
            "fsq": float(row.fsq),
            "normalized_force": float(row.normalized_force),
            "step_size": float(row.step_size),
            "accepted": bool(row.accepted),
        }
        for row in result.trace
    ]


def _norm(values) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.sum(values**2)))


def _edge_slices(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return cap, cap-adjacent, and interior axial slices for a residual array."""
    if values.shape[-1] == 1:
        return values, values[..., :0], values[..., :0]
    caps = np.concatenate((values[..., :1], values[..., -1:]), axis=-1)
    if values.shape[-1] <= 2:
        return caps, values[..., :0], values[..., :0]
    adjacent = np.concatenate((values[..., 1:2], values[..., -2:-1]), axis=-1)
    interior = values[..., 2:-2] if values.shape[-1] > 4 else values[..., :0]
    return caps, adjacent, interior


def _residual_component_metrics(result) -> dict[str, object]:
    residual = axisym_projected_energy_residual(
        result.state,
        result.grid,
        psi_prime=result.psi_prime,
        i_prime=result.i_prime,
        pressure=result.pressure,
        mu0=result.options.mu0,
    )
    projected_a = np.asarray(residual.projected_a, dtype=float)
    projected_lam = np.asarray(residual.projected_lam, dtype=float)
    a_caps, a_cap_adjacent, a_interior = _edge_slices(projected_a)
    lam_caps, lam_cap_adjacent, lam_interior = _edge_slices(projected_lam)
    radial_axis = projected_lam[:1, :]
    radial_edge = projected_lam[-1:, :]
    radial_interior = projected_lam[1:-1, :] if projected_lam.shape[0] > 2 else projected_lam[:0, :]
    total = max(float(residual.norm), np.finfo(float).tiny)
    a_norm = _norm(projected_a)
    lam_norm = _norm(projected_lam)
    return {
        "component_energy": float(residual.energy),
        "component_norm": float(residual.norm),
        "component_fsq": float(residual.fsq),
        "component_normalized_force": float(residual.normalized_force),
        "component_active_dof": int(residual.active_dof),
        "residual_a_norm": a_norm,
        "residual_lam_norm": lam_norm,
        "residual_a_fraction": float(a_norm / total),
        "residual_lam_fraction": float(lam_norm / total),
        "residual_a_cap_norm": _norm(a_caps),
        "residual_a_cap_adjacent_norm": _norm(a_cap_adjacent),
        "residual_a_interior_xi_norm": _norm(a_interior),
        "residual_lam_cap_norm": _norm(lam_caps),
        "residual_lam_cap_adjacent_norm": _norm(lam_cap_adjacent),
        "residual_lam_interior_xi_norm": _norm(lam_interior),
        "residual_lam_radial_axis_norm": _norm(radial_axis),
        "residual_lam_radial_edge_norm": _norm(radial_edge),
        "residual_lam_radial_interior_norm": _norm(radial_interior),
        "residual_a_max_abs": float(np.max(np.abs(projected_a))) if projected_a.size else 0.0,
        "residual_lam_max_abs": float(np.max(np.abs(projected_lam))) if projected_lam.size else 0.0,
    }


def _run_one(
    *,
    ns: int,
    nxi: int,
    maxiter: int,
    residual_linear_maxiter: int,
    residual_linear_maxiter_policy: str,
    residual_linear_adaptive_factor: float,
    residual_linear_solver: str,
    preconditioner: str,
    residual_radial_alpha: float,
    residual_lambda_alpha: float,
    residual_xi_alpha: float,
    line_search_steps: int,
    gtol: float,
    ftol: float,
    coil_radius: float,
    separation: float,
    current: float,
    midplane_radius: float,
    i_prime: float,
    perturbation: float,
):
    problem = _two_coil_problem(
        ns=ns,
        nxi=nxi,
        coil_radius=coil_radius,
        separation=separation,
        current=current,
        midplane_radius=midplane_radius,
        i_prime=i_prime,
        perturbation=perturbation,
    )
    result = run_mirror_fixed_boundary(
        problem["config"],
        problem["boundary"],
        psi_prime=problem["psi_prime"],
        i_prime=problem["i_prime"],
        pressure=problem["pressure"],
        initial_state=problem["initial_state"],
        options=MirrorSolveOptions(
            optimizer="residual_newton",
            maxiter=maxiter,
            tolerance=gtol,
            ftol=ftol,
            line_search_steps=line_search_steps,
            residual_linear_maxiter=residual_linear_maxiter,
            residual_linear_maxiter_policy=residual_linear_maxiter_policy,
            residual_linear_adaptive_factor=residual_linear_adaptive_factor,
            residual_linear_solver=residual_linear_solver,
            residual_preconditioner=preconditioner,
            residual_radial_alpha=residual_radial_alpha,
            residual_lambda_alpha=residual_lambda_alpha,
            residual_xi_alpha=residual_xi_alpha,
            mu0=1.0,
        ),
    )
    row_id = f"ns{ns}_nxi{nxi}_outer{maxiter}_linear{residual_linear_maxiter}_{residual_linear_solver}_{preconditioner}"
    summary = result.optimizer_summaries[-1] if result.optimizer_summaries else None
    first = result.trace[0]
    final = result.final_trace
    row = {
        "row_id": row_id,
        "ns": int(ns),
        "nxi": int(nxi),
        "optimizer": "residual_newton",
        "maxiter": int(maxiter),
        "line_search_steps": int(line_search_steps),
        "residual_linear_maxiter": int(residual_linear_maxiter),
        "residual_linear_maxiter_policy": str(residual_linear_maxiter_policy),
        "residual_linear_adaptive_factor": float(residual_linear_adaptive_factor),
        "residual_linear_solver": str(residual_linear_solver),
        "residual_linear_maxiter_effective_max": None
        if summary is None
        else summary.residual_linear_maxiter_effective_max,
        "residual_linear_maxiter_effective_last": None
        if summary is None
        else summary.residual_linear_maxiter_effective_last,
        "residual_preconditioner": str(preconditioner),
        "residual_radial_alpha": float(residual_radial_alpha),
        "residual_lambda_alpha": float(residual_lambda_alpha),
        "residual_xi_alpha": float(residual_xi_alpha),
        "gtol": float(gtol),
        "ftol": float(ftol),
        "psi_value": float(problem["psi_value"]),
        "i_prime_value": float(problem["i_prime_value"]),
        "twist_proxy_i_prime_over_psi_prime": float(problem["twist_proxy_i_prime_over_psi_prime"]),
        "finite_current": bool(abs(float(problem["i_prime_value"])) > 0.0),
        "trace_steps": int(len(result.trace)),
        "optimizer_nit": int(summary.nit) if summary is not None else 0,
        "optimizer_nfev": int(summary.nfev) if summary is not None else 0,
        "optimizer_njev": int(summary.njev) if summary is not None else 0,
        "optimizer_success": bool(summary.success) if summary is not None else False,
        "optimizer_accepted": bool(summary.accepted) if summary is not None else False,
        "optimizer_status": int(summary.status) if summary is not None else -1,
        "optimizer_message": str(summary.message) if summary is not None else "",
        "optimizer_rejection_reason": str(summary.rejection_reason) if summary is not None else "",
        "initial_energy_total": float(first.energy_total),
        "final_energy_total": float(final.energy_total),
        "energy_drop": float(first.energy_total - final.energy_total),
        "initial_residual_norm": float(first.residual_norm),
        "final_residual_norm": float(final.residual_norm),
        "residual_drop": float(first.residual_norm - final.residual_norm),
        "residual_reduction_factor": float(final.residual_norm / max(first.residual_norm, np.finfo(float).tiny)),
        "final_fsq": float(final.fsq),
        "final_normalized_force": float(final.normalized_force),
        "reached_projected_gtol": bool(final.residual_norm <= float(gtol)),
        "min_sqrtg": float(final.min_sqrtg),
        "mirror_ratio": float(final.mirror_ratio),
    }
    row.update(_residual_component_metrics(result))
    return row, _history_rows(result, row_id=row_id), result


def _rows_for_filter(rows, **kwargs):
    return [row for row in rows if all(row[key] == value for key, value in kwargs.items())]


def _write_resolution_heatmap(
    rows: list[dict[str, object]],
    *,
    outdir: Path,
    preconditioner: str,
    maxiter: int,
    residual_linear_maxiter: int,
) -> Path | None:
    import matplotlib.pyplot as plt

    filtered = _rows_for_filter(
        rows,
        residual_preconditioner=preconditioner,
        maxiter=maxiter,
        residual_linear_maxiter=residual_linear_maxiter,
    )
    if not filtered:
        return None
    ns_values = sorted({int(row["ns"]) for row in filtered})
    nxi_values = sorted({int(row["nxi"]) for row in filtered})
    values = np.full((len(ns_values), len(nxi_values)), np.nan)
    for row in filtered:
        i = ns_values.index(int(row["ns"]))
        j = nxi_values.index(int(row["nxi"]))
        values[i, j] = max(float(row["final_residual_norm"]), np.finfo(float).tiny)

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    image = ax.imshow(np.log10(values), origin="lower", aspect="auto", cmap="viridis_r")
    ax.set_xticks(np.arange(len(nxi_values)))
    ax.set_xticklabels([str(value) for value in nxi_values])
    ax.set_yticks(np.arange(len(ns_values)))
    ax.set_yticklabels([str(value) for value in ns_values])
    ax.set_xlabel("nxi")
    ax.set_ylabel("ns")
    ax.set_title(f"log10 final residual, {preconditioner}")
    for i in range(len(ns_values)):
        for j in range(len(nxi_values)):
            if np.isfinite(values[i, j]):
                ax.text(j, i, f"{values[i, j]:.1e}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=ax, label="log10 residual")
    fig.tight_layout()
    path = outdir / "residual_newton_convergence_resolution_heatmap.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_budget_plot(
    rows: list[dict[str, object]],
    *,
    outdir: Path,
    preconditioner: str,
    ns: int,
    nxi: int,
) -> Path | None:
    import matplotlib.pyplot as plt

    filtered = _rows_for_filter(rows, residual_preconditioner=preconditioner, ns=ns, nxi=nxi)
    if not filtered:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for linear in sorted({int(row["residual_linear_maxiter"]) for row in filtered}):
        items = sorted(
            [row for row in filtered if int(row["residual_linear_maxiter"]) == linear],
            key=lambda row: int(row["maxiter"]),
        )
        ax.semilogy(
            [int(row["maxiter"]) for row in items],
            [float(row["final_residual_norm"]) for row in items],
            "o-",
            label=f"linear maxiter {linear}",
        )
    ax.set_xlabel("outer maxiter")
    ax.set_ylabel("final projected residual")
    ax.set_title(f"two-coil residual Newton budgets, ns={ns}, nxi={nxi}")
    ax.grid(True, which="both", linewidth=0.35, alpha=0.35)
    ax.legend(fontsize="small")
    fig.tight_layout()
    path = outdir / "residual_newton_convergence_budget.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_preconditioner_plot(
    rows: list[dict[str, object]],
    *,
    outdir: Path,
    ns: int,
    nxi: int,
    maxiter: int,
    residual_linear_maxiter: int,
) -> Path | None:
    import matplotlib.pyplot as plt

    filtered = _rows_for_filter(
        rows,
        ns=ns,
        nxi=nxi,
        maxiter=maxiter,
        residual_linear_maxiter=residual_linear_maxiter,
    )
    if len(filtered) < 2:
        return None
    filtered = sorted(filtered, key=lambda row: str(row["residual_preconditioner"]))
    labels = [str(row["residual_preconditioner"]) for row in filtered]
    values = [max(float(row["final_residual_norm"]), np.finfo(float).tiny) for row in filtered]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.bar(np.arange(len(values)), values, color=["0.45", "tab:green", "tab:red"][: len(values)])
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("final projected residual")
    ax.set_title(f"preconditioner comparison, ns={ns}, nxi={nxi}")
    ax.grid(True, which="both", axis="y", linewidth=0.35, alpha=0.35)
    fig.tight_layout()
    path = outdir / "residual_newton_convergence_preconditioners.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_history_plot(
    rows: list[dict[str, object]],
    histories: list[dict[str, object]],
    *,
    outdir: Path,
    ns: int,
    nxi: int,
    maxiter: int,
    residual_linear_maxiter: int,
) -> Path | None:
    import matplotlib.pyplot as plt

    row_ids = {
        str(row["row_id"]): row
        for row in rows
        if int(row["ns"]) == ns
        and int(row["nxi"]) == nxi
        and int(row["maxiter"]) == maxiter
        and int(row["residual_linear_maxiter"]) == residual_linear_maxiter
    }
    if not row_ids:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for history in histories:
        row = row_ids.get(str(history["row_id"]))
        if row is None:
            continue
        trace = history["history"]
        ax.semilogy(
            np.arange(len(trace)),
            [float(item["residual_norm"]) for item in trace],
            "o-",
            label=str(row["residual_preconditioner"]),
        )
    ax.set_xlabel("recorded trace index")
    ax.set_ylabel("projected residual")
    ax.set_title(f"residual history, ns={ns}, nxi={nxi}")
    ax.grid(True, which="both", linewidth=0.35, alpha=0.35)
    ax.legend(fontsize="small")
    fig.tight_layout()
    path = outdir / "residual_newton_convergence_history.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_component_plot(
    rows: list[dict[str, object]],
    *,
    outdir: Path,
    ns: int,
    nxi: int,
    maxiter: int,
    residual_linear_maxiter: int,
) -> Path | None:
    import matplotlib.pyplot as plt

    filtered = _rows_for_filter(
        rows,
        ns=ns,
        nxi=nxi,
        maxiter=maxiter,
        residual_linear_maxiter=residual_linear_maxiter,
    )
    if not filtered:
        return None
    filtered = sorted(filtered, key=lambda row: str(row["residual_preconditioner"]))
    labels = [str(row["residual_preconditioner"]) for row in filtered]
    component_keys = (
        ("a adjacent caps", "residual_a_cap_adjacent_norm"),
        ("a interior xi", "residual_a_interior_xi_norm"),
        ("lambda caps", "residual_lam_cap_norm"),
        ("lambda adjacent caps", "residual_lam_cap_adjacent_norm"),
        ("lambda interior xi", "residual_lam_interior_xi_norm"),
    )
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    colors = ("tab:blue", "tab:cyan", "tab:orange", "tab:red", "tab:green")
    x = np.arange(len(filtered), dtype=float)
    offsets = np.linspace(-0.3, 0.3, len(component_keys))
    width = 0.12 if len(filtered) > 1 else 0.08
    plotted_components = 0
    for (label, key), color, offset in zip(component_keys, colors, offsets, strict=True):
        values = np.asarray([max(float(row[key]), 0.0) for row in filtered], dtype=float)
        positive = values > 0.0
        if not np.any(positive):
            continue
        ax.bar(x[positive] + offset, values[positive], width=width, label=label, color=color)
        plotted_components += 1
    totals = np.asarray([max(float(row["final_residual_norm"]), np.finfo(float).tiny) for row in filtered], dtype=float)
    ax.plot(x, totals, "ko", ms=4, label="total norm")
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("projected residual component norm")
    ax.set_title(f"residual components, ns={ns}, nxi={nxi}")
    ax.grid(True, which="both", axis="y", linewidth=0.35, alpha=0.35)
    ax.legend(fontsize="x-small", ncols=2 if plotted_components > 2 else 1)
    fig.tight_layout()
    path = outdir / "residual_newton_convergence_components.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_selected_output(result, *, outdir: Path, label: str) -> dict[str, str]:
    case_dir = outdir / label
    mout = write_mirror_output(case_dir / f"mout_{label}.nc", result, overwrite=True)
    output = load_mirror_output(mout)
    figure_dir = case_dir / "figures"
    plot_mirror_output(output, outdir=figure_dir, name=label)
    return {"mout": str(mout), "figures": str(figure_dir)}


def _write_plots(
    *,
    rows: list[dict[str, object]],
    histories: list[dict[str, object]],
    best_result,
    reference_result,
    outdir: Path,
    case_label: str,
) -> tuple[list[str], dict[str, dict[str, str]]]:
    outdir.mkdir(parents=True, exist_ok=True)
    default_preconditioner = "radial_xi_tridi"
    if not any(row["residual_preconditioner"] == default_preconditioner for row in rows):
        default_preconditioner = str(rows[0]["residual_preconditioner"])
    max_ns = max(int(row["ns"]) for row in rows)
    max_nxi = max(int(row["nxi"]) for row in rows)
    max_outer = max(int(row["maxiter"]) for row in rows)
    max_linear = max(int(row["residual_linear_maxiter"]) for row in rows)

    figures: list[str] = []
    for path in (
        _write_resolution_heatmap(
            rows,
            outdir=outdir,
            preconditioner=default_preconditioner,
            maxiter=max_outer,
            residual_linear_maxiter=max_linear,
        ),
        _write_budget_plot(rows, outdir=outdir, preconditioner=default_preconditioner, ns=max_ns, nxi=max_nxi),
        _write_preconditioner_plot(
            rows,
            outdir=outdir,
            ns=max_ns,
            nxi=max_nxi,
            maxiter=max_outer,
            residual_linear_maxiter=max_linear,
        ),
        _write_history_plot(
            rows,
            histories,
            outdir=outdir,
            ns=max_ns,
            nxi=max_nxi,
            maxiter=max_outer,
            residual_linear_maxiter=max_linear,
        ),
        _write_component_plot(
            rows,
            outdir=outdir,
            ns=max_ns,
            nxi=max_nxi,
            maxiter=max_outer,
            residual_linear_maxiter=max_linear,
        ),
    ):
        if path is not None:
            figures.append(str(path))
    selected_artifacts = {
        "best_residual": _write_selected_output(best_result, outdir=outdir, label=f"best_{case_label}_residual_newton")
    }
    if reference_result is not None and reference_result is not best_result:
        selected_artifacts["highest_budget"] = _write_selected_output(
            reference_result,
            outdir=outdir,
            label=f"highest_budget_{case_label}_residual_newton",
        )
    return figures, selected_artifacts


def run_case(
    outdir: Path,
    *,
    ns_array: tuple[int, ...] = (5, 9),
    nxi_array: tuple[int, ...] = (9, 17),
    maxiter_array: tuple[int, ...] = (6, 12),
    residual_linear_maxiter_array: tuple[int, ...] = (16, 48),
    residual_linear_maxiter_policy: str = "fixed",
    residual_linear_adaptive_factor: float = 6.0,
    residual_linear_solver: str = "lsmr",
    preconditioners: tuple[str, ...] = ("radial_xi_tridi",),
    residual_radial_alpha: float = 0.5,
    residual_lambda_alpha: float = 0.5,
    residual_xi_alpha: float = 0.2,
    line_search_steps: int = 32,
    gtol: float = 1.0e-12,
    ftol: float = 1.0e-12,
    coil_radius: float = 0.35,
    separation: float = 2.0,
    current: float = 1.0e6,
    midplane_radius: float = 0.3,
    i_prime: float = 0.0,
    case_label: str = "",
    perturbation: float = 0.02,
    write_plots: bool = True,
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    selected_case_label = _safe_label(case_label or _default_case_label(i_prime))
    rows: list[dict[str, object]] = []
    histories: list[dict[str, object]] = []
    best_result = None
    reference_result = None
    best_residual = np.inf
    reference_selector = {
        "preconditioner": "radial_xi_tridi" if "radial_xi_tridi" in preconditioners else preconditioners[0],
        "ns": max(ns_array),
        "nxi": max(nxi_array),
        "maxiter": max(maxiter_array),
        "residual_linear_maxiter": max(residual_linear_maxiter_array),
    }

    for preconditioner in preconditioners:
        for ns in ns_array:
            for nxi in nxi_array:
                for maxiter in maxiter_array:
                    for residual_linear_maxiter in residual_linear_maxiter_array:
                        row, history, result = _run_one(
                            ns=ns,
                            nxi=nxi,
                            maxiter=maxiter,
                            residual_linear_maxiter=residual_linear_maxiter,
                            residual_linear_maxiter_policy=residual_linear_maxiter_policy,
                            residual_linear_adaptive_factor=residual_linear_adaptive_factor,
                            residual_linear_solver=residual_linear_solver,
                            preconditioner=preconditioner,
                            residual_radial_alpha=residual_radial_alpha,
                            residual_lambda_alpha=residual_lambda_alpha,
                            residual_xi_alpha=residual_xi_alpha,
                            line_search_steps=line_search_steps,
                            gtol=gtol,
                            ftol=ftol,
                            coil_radius=coil_radius,
                            separation=separation,
                            current=current,
                            midplane_radius=midplane_radius,
                            i_prime=i_prime,
                            perturbation=perturbation,
                        )
                        rows.append(row)
                        histories.append({"row_id": row["row_id"], "history": history})
                        if float(row["final_residual_norm"]) < best_residual:
                            best_residual = float(row["final_residual_norm"])
                            best_result = result
                        if (
                            preconditioner == reference_selector["preconditioner"]
                            and int(ns) == int(reference_selector["ns"])
                            and int(nxi) == int(reference_selector["nxi"])
                            and int(maxiter) == int(reference_selector["maxiter"])
                            and int(residual_linear_maxiter) == int(reference_selector["residual_linear_maxiter"])
                        ):
                            reference_result = result

    figures: list[str] = []
    selected_artifacts: dict[str, dict[str, str]] = {}
    if write_plots and best_result is not None:
        figures, selected_artifacts = _write_plots(
            rows=rows,
            histories=histories,
            best_result=best_result,
            reference_result=reference_result,
            outdir=outdir,
            case_label=selected_case_label,
        )

    payload = {
        "case_label": selected_case_label,
        "i_prime_value": float(i_prime),
        "rows": rows,
        "histories": histories,
        "figures": figures,
        "selected_artifacts": selected_artifacts,
    }
    path = outdir / "residual_newton_convergence_grid_metrics.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = run_case(
        args.outdir,
        ns_array=_parse_int_array(args.ns_array, name="ns-array", minimum=3),
        nxi_array=_parse_int_array(args.nxi_array, name="nxi-array", minimum=3),
        maxiter_array=_parse_int_array(args.maxiter_array, name="maxiter-array", minimum=1),
        residual_linear_maxiter_array=_parse_int_array(
            args.residual_linear_maxiter_array,
            name="residual-linear-maxiter-array",
            minimum=1,
        ),
        residual_linear_maxiter_policy=args.residual_linear_maxiter_policy,
        residual_linear_adaptive_factor=args.residual_linear_adaptive_factor,
        residual_linear_solver=args.residual_linear_solver,
        preconditioners=_parse_preconditioners(args.preconditioners),
        residual_radial_alpha=args.residual_radial_alpha,
        residual_lambda_alpha=args.residual_lambda_alpha,
        residual_xi_alpha=args.residual_xi_alpha,
        line_search_steps=args.line_search_steps,
        gtol=args.gtol,
        ftol=args.ftol,
        coil_radius=args.coil_radius,
        separation=args.separation,
        current=args.current,
        midplane_radius=args.midplane_radius,
        i_prime=args.i_prime,
        case_label=args.case_label,
        perturbation=args.perturbation,
        write_plots=not args.no_plots,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
