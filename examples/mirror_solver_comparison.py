"""Compare mirror fixed-boundary solver paths on small axisymmetric cases."""

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
    MirrorBoundary,
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
from vmec_jax.mirror.validation.manufactured import make_mms_case, solve_axisym_mms_fixed_boundary


PRODUCTION_OPTIMIZERS = ("gradient_descent", "lbfgs", "residual_newton")
SUPPORTED_CASES = ("cylinder", "two_coil", "manufactured")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("results/mirror/solver_comparison"))
    parser.add_argument("--cases", type=str, default="cylinder,two_coil,manufactured")
    parser.add_argument("--maxiter-gd", type=int, default=12)
    parser.add_argument("--maxiter-lbfgs", type=int, default=40)
    parser.add_argument("--maxiter-newton", type=int, default=12)
    parser.add_argument("--line-search-steps", type=int, default=32)
    parser.add_argument("--residual-linear-maxiter", type=int, default=48)
    parser.add_argument(
        "--residual-linear-maxiter-policy",
        type=str,
        default="adaptive",
        choices=("fixed", "adaptive"),
    )
    parser.add_argument("--residual-linear-adaptive-factor", type=float, default=6.0)
    parser.add_argument(
        "--residual-linear-solver",
        type=str,
        default="lsmr",
        choices=("lsmr", "dense_lstsq"),
        help="Linear solver for residual-Newton corrections.",
    )
    parser.add_argument(
        "--residual-preconditioner",
        type=str,
        default="radial_xi_tridi",
        choices=("none", "radial_tridi", "radial_xi_tridi", "radial_xi_lambda_xi_tridi"),
    )
    parser.add_argument("--residual-radial-alpha", type=float, default=0.5)
    parser.add_argument("--residual-lambda-alpha", type=float, default=0.5)
    parser.add_argument("--residual-xi-alpha", type=float, default=0.2)
    parser.add_argument("--gtol", type=float, default=1.0e-12)
    parser.add_argument("--ftol", type=float, default=1.0e-12)
    parser.add_argument("--cylinder-ns", type=int, default=5)
    parser.add_argument("--cylinder-nxi", type=int, default=9)
    parser.add_argument("--cylinder-radius", type=float, default=0.3)
    parser.add_argument("--cylinder-perturbation", type=float, default=0.01)
    parser.add_argument("--two-coil-ns", type=int, default=9)
    parser.add_argument("--two-coil-nxi", type=int, default=17)
    parser.add_argument("--coil-radius", type=float, default=0.35)
    parser.add_argument("--separation", type=float, default=2.0)
    parser.add_argument("--current", type=float, default=1.0e6)
    parser.add_argument("--midplane-radius", type=float, default=0.3)
    parser.add_argument("--two-coil-perturbation", type=float, default=0.02)
    parser.add_argument("--manufactured-ns", type=int, default=5)
    parser.add_argument("--manufactured-nxi", type=int, default=9)
    parser.add_argument("--manufactured-perturbation", type=float, default=0.002)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def _parse_cases(value: str) -> tuple[str, ...]:
    cases = tuple(item.strip().lower().replace("-", "_") for item in str(value).split(",") if item.strip())
    if not cases:
        raise ValueError("--cases must contain at least one case name")
    unknown = sorted(set(cases) - set(SUPPORTED_CASES))
    if unknown:
        raise ValueError(f"unsupported case name(s): {', '.join(unknown)}")
    return cases


def _optimizer_maxiter(optimizer: str, *, maxiter_gd: int, maxiter_lbfgs: int, maxiter_newton: int) -> int:
    if optimizer == "gradient_descent":
        return int(maxiter_gd)
    if optimizer == "lbfgs":
        return int(maxiter_lbfgs)
    if optimizer == "residual_newton":
        return int(maxiter_newton)
    raise ValueError(f"unsupported optimizer {optimizer!r}")


def _perturbed_initial_state(config: MirrorConfig, boundary: MirrorBoundary, *, amplitude: float) -> MirrorStateAxisym:
    grid = config.build_grid()
    base = MirrorStateAxisym.from_boundary(grid, boundary)
    s = grid.s_full[:, None]
    xi = grid.xi[None, :]
    shape = s * (1.0 - s) * (1.0 - xi**2)
    a = base.a * (1.0 + float(amplitude) * shape)
    return MirrorStateAxisym(a=a, lam=np.zeros_like(a))


def _cylinder_problem(*, ns: int, nxi: int, radius: float, perturbation: float) -> dict[str, object]:
    config = MirrorConfig(MirrorResolution(ns=int(ns), ntheta=1, nxi=int(nxi), mpol=0), z_min=-1.0, z_max=1.0)
    boundary = MirrorBoundary.constant_radius(float(radius))
    return {
        "name": "cylinder",
        "config": config,
        "boundary": boundary,
        "initial_state": _perturbed_initial_state(config, boundary, amplitude=perturbation),
        "psi_prime": PsiPrimeProfile.constant(0.01),
        "i_prime": IPrimeProfile.zero(),
        "pressure": PressureProfile.zero(),
        "step_size": 0.05,
        "metadata": {"radius": float(radius), "perturbation": float(perturbation)},
    }


def _two_coil_problem(
    *,
    ns: int,
    nxi: int,
    coil_radius: float,
    separation: float,
    current: float,
    midplane_radius: float,
    perturbation: float,
) -> dict[str, object]:
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
        "name": "two_coil",
        "config": config,
        "boundary": boundary,
        "initial_state": _perturbed_initial_state(config, boundary, amplitude=perturbation),
        "psi_prime": PsiPrimeProfile.constant(psi_value),
        "i_prime": IPrimeProfile.zero(),
        "pressure": PressureProfile.zero(),
        "step_size": 0.01,
        "metadata": {
            "coil_radius": float(coil_radius),
            "separation": float(separation),
            "current": float(current),
            "midplane_radius": float(midplane_radius),
            "psi_value": float(psi_value),
            "perturbation": float(perturbation),
        },
    }


def _history_from_fixed_boundary_result(result) -> list[dict[str, object]]:
    return [
        {
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


def _history_from_mms_result(result) -> list[dict[str, object]]:
    return [
        {
            "iteration": int(row.iteration),
            "objective": float(row.objective),
            "residual_norm": float(row.residual_norm),
            "fsq": float(row.fsq),
            "exact_error_norm": float(row.exact_error_norm),
            "step_norm": float(row.step_norm),
        }
        for row in result.trace
    ]


def _fixed_boundary_row(
    *,
    case_name: str,
    optimizer: str,
    result,
    maxiter: int,
    gtol: float,
    ftol: float,
) -> dict[str, object]:
    summary = result.optimizer_summaries[-1] if result.optimizer_summaries else None
    first = result.trace[0]
    final = result.final_trace
    reduction = float(final.residual_norm / max(first.residual_norm, np.finfo(float).tiny))
    return {
        "case": str(case_name),
        "solver_scope": "production_fixed_boundary",
        "optimizer": str(optimizer),
        "residual_linear_maxiter": int(result.options.residual_linear_maxiter),
        "residual_linear_maxiter_policy": str(result.options.residual_linear_maxiter_policy),
        "residual_linear_adaptive_factor": float(result.options.residual_linear_adaptive_factor),
        "residual_linear_solver": str(result.options.residual_linear_solver),
        "residual_linear_maxiter_effective_max": None
        if summary is None
        else summary.residual_linear_maxiter_effective_max,
        "residual_linear_maxiter_effective_last": None
        if summary is None
        else summary.residual_linear_maxiter_effective_last,
        "residual_preconditioner": str(result.options.residual_preconditioner),
        "residual_radial_alpha": float(result.options.residual_radial_alpha),
        "residual_lambda_alpha": float(result.options.residual_lambda_alpha),
        "residual_xi_alpha": float(result.options.residual_xi_alpha),
        "maxiter": int(maxiter),
        "gtol": float(gtol),
        "ftol": float(ftol),
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
        "residual_reduction_factor": reduction,
        "final_fsq": float(final.fsq),
        "final_normalized_force": float(final.normalized_force),
        "reached_projected_gtol": bool(final.residual_norm <= float(gtol)),
        "min_sqrtg": float(final.min_sqrtg),
        "mirror_ratio": float(final.mirror_ratio),
    }


def _run_fixed_boundary_case(
    problem: dict[str, object],
    *,
    maxiter_gd: int,
    maxiter_lbfgs: int,
    maxiter_newton: int,
    line_search_steps: int,
    residual_linear_maxiter: int,
    residual_linear_maxiter_policy: str,
    residual_linear_adaptive_factor: float,
    residual_linear_solver: str,
    residual_preconditioner: str,
    residual_radial_alpha: float,
    residual_lambda_alpha: float,
    residual_xi_alpha: float,
    gtol: float,
    ftol: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    histories: list[dict[str, object]] = []
    residual_newton_result = None
    for optimizer in PRODUCTION_OPTIMIZERS:
        maxiter = _optimizer_maxiter(
            optimizer,
            maxiter_gd=maxiter_gd,
            maxiter_lbfgs=maxiter_lbfgs,
            maxiter_newton=maxiter_newton,
        )
        result = run_mirror_fixed_boundary(
            problem["config"],
            problem["boundary"],
            psi_prime=problem["psi_prime"],
            i_prime=problem["i_prime"],
            pressure=problem["pressure"],
            initial_state=problem["initial_state"],
            options=MirrorSolveOptions(
                optimizer=optimizer,
                maxiter=maxiter,
                tolerance=gtol,
                ftol=ftol,
                step_size=problem["step_size"],
                line_search_steps=line_search_steps,
                residual_linear_maxiter=residual_linear_maxiter,
                residual_linear_maxiter_policy=residual_linear_maxiter_policy,
                residual_linear_adaptive_factor=residual_linear_adaptive_factor,
                residual_linear_solver=residual_linear_solver,
                residual_preconditioner=residual_preconditioner,
                residual_radial_alpha=residual_radial_alpha,
                residual_lambda_alpha=residual_lambda_alpha,
                residual_xi_alpha=residual_xi_alpha,
                mu0=1.0,
            ),
        )
        row = _fixed_boundary_row(
            case_name=problem["name"],
            optimizer=optimizer,
            result=result,
            maxiter=maxiter,
            gtol=gtol,
            ftol=ftol,
        )
        rows.append(row)
        histories.append(
            {
                "case": str(problem["name"]),
                "optimizer": optimizer,
                "solver_scope": "production_fixed_boundary",
                "history": _history_from_fixed_boundary_result(result),
            }
        )
        if optimizer == "residual_newton":
            residual_newton_result = result
    selected = {"residual_newton": residual_newton_result}
    return rows, histories, selected


def _manufactured_initial_state(case, *, amplitude: float) -> MirrorStateAxisym:
    grid = case.grid
    s = grid.s_full[:, None]
    xi = grid.xi[None, :]
    shape = s * (1.0 - s) * (1.0 - xi**2)
    a = case.state.a * (1.0 + float(amplitude) * shape)
    lam = case.state.lam + 0.1 * float(amplitude) * shape * xi
    return MirrorStateAxisym(a=a, lam=lam)


def _run_manufactured_case(
    *,
    ns: int,
    nxi: int,
    maxiter: int,
    line_search_steps: int,
    gtol: float,
    ftol: float,
    perturbation: float,
) -> tuple[dict[str, object], dict[str, object]]:
    case = make_mms_case("axisym_projected_fixed_boundary", MirrorResolution(ns=ns, ntheta=1, nxi=nxi, mpol=0), mu0=1.0)
    result = solve_axisym_mms_fixed_boundary(
        case,
        initial_state=_manufactured_initial_state(case, amplitude=perturbation),
        maxiter=maxiter,
        gtol=gtol,
        ftol=ftol,
        line_search_steps=line_search_steps,
        mu0=1.0,
    )
    first = result.trace[0]
    reduction = float(result.residual_norm / max(first.residual_norm, np.finfo(float).tiny))
    row = {
        "case": "manufactured",
        "solver_scope": "manufactured_source_validation",
        "optimizer": "residual_newton",
        "maxiter": int(maxiter),
        "gtol": float(gtol),
        "ftol": float(ftol),
        "trace_steps": int(len(result.trace)),
        "optimizer_nit": int(result.optimizer_nit),
        "optimizer_nfev": int(result.optimizer_nfev),
        "optimizer_njev": int(result.optimizer_njev),
        "optimizer_success": bool(result.optimizer_success),
        "optimizer_accepted": bool(result.optimizer_success),
        "optimizer_status": int(result.optimizer_status),
        "optimizer_message": result.optimizer_message,
        "optimizer_rejection_reason": "",
        "initial_energy_total": float(first.objective),
        "final_energy_total": float(result.final_trace.objective),
        "energy_drop": float(first.objective - result.final_trace.objective),
        "initial_residual_norm": float(first.residual_norm),
        "final_residual_norm": float(result.residual_norm),
        "residual_drop": float(first.residual_norm - result.residual_norm),
        "residual_reduction_factor": reduction,
        "final_fsq": float(result.fsq),
        "final_normalized_force": float(result.residual_norm),
        "reached_projected_gtol": bool(result.residual_norm <= float(gtol)),
        "final_exact_error_norm": float(result.exact_error_norm),
        "initial_exact_error_norm": float(first.exact_error_norm),
        "min_sqrtg": None,
        "mirror_ratio": None,
        "note": "sourced MMS validation solve; production gradient_descent/lbfgs do not yet include MMS sources",
    }
    history = {
        "case": "manufactured",
        "optimizer": "residual_newton",
        "solver_scope": "manufactured_source_validation",
        "history": _history_from_mms_result(result),
    }
    return row, history


def _write_residual_history_plot(histories: list[dict[str, object]], *, outdir: Path) -> Path:
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    cases = [case for case in SUPPORTED_CASES if any(history["case"] == case for history in histories)]
    fig, axes = plt.subplots(len(cases), 1, figsize=(7.2, 3.1 * len(cases)), sharex=False)
    if len(cases) == 1:
        axes = [axes]
    for ax, case in zip(axes, cases, strict=True):
        for history in histories:
            if history["case"] != case:
                continue
            rows = history["history"]
            x = np.arange(len(rows), dtype=int)
            residual = np.asarray([row["residual_norm"] for row in rows], dtype=float)
            label = str(history["optimizer"])
            if history["solver_scope"] == "manufactured_source_validation":
                label = f"{label} (MMS source)"
            ax.semilogy(x, residual, "o-", markersize=3.5, label=label)
        ax.set_ylabel("projected residual")
        ax.set_title(f"{case.replace('_', '-')} convergence")
        ax.grid(True, which="both", linewidth=0.35, alpha=0.35)
        ax.legend(fontsize="small")
    axes[-1].set_xlabel("recorded trace index")
    fig.tight_layout()
    path = outdir / "solver_comparison_residuals.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_final_residual_plot(rows: list[dict[str, object]], *, outdir: Path) -> Path:
    import matplotlib.pyplot as plt

    labels = []
    values = []
    colors = []
    palette = {
        "gradient_descent": "tab:blue",
        "lbfgs": "tab:green",
        "residual_newton": "tab:red",
    }
    for row in rows:
        suffix = "MMS" if row["solver_scope"] == "manufactured_source_validation" else str(row["optimizer"])
        labels.append(f"{row['case']}\n{suffix}")
        values.append(max(float(row["final_residual_norm"]), np.finfo(float).tiny))
        colors.append(palette.get(str(row["optimizer"]), "0.5"))

    fig, ax = plt.subplots(figsize=(max(7.0, 0.82 * len(labels)), 4.0))
    ax.bar(np.arange(len(values)), values, color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("final projected residual")
    ax.set_title("solver comparison final residuals")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(True, which="both", axis="y", linewidth=0.35, alpha=0.35)
    fig.tight_layout()
    path = outdir / "solver_comparison_final_residuals.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_boundary_plot(problems: list[dict[str, object]], *, outdir: Path) -> Path | None:
    import matplotlib.pyplot as plt

    if not problems:
        return None
    fig, ax = plt.subplots(figsize=(7.0, 3.75))
    for problem in problems:
        config = problem["config"]
        boundary = problem["boundary"]
        grid = config.build_grid()
        radius = boundary.radius_on_grid(grid)
        ax.plot(grid.z, radius, linewidth=1.8, label=str(problem["name"]))
        ax.plot(grid.z, -radius, linewidth=1.8, color=ax.lines[-1].get_color())
    ax.set_xlabel("z")
    ax.set_ylabel("x")
    ax.set_title("physical benchmark boundaries")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.35, alpha=0.35)
    ax.legend(fontsize="small")
    fig.tight_layout()
    path = outdir / "solver_comparison_boundaries.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_selected_physical_outputs(
    selected_results: dict[str, object],
    *,
    outdir: Path,
    write_plots: bool,
) -> list[dict[str, str]]:
    artifacts: list[dict[str, str]] = []
    if not write_plots:
        return artifacts
    for case_name, result in selected_results.items():
        if result is None:
            continue
        case_dir = outdir / f"{case_name}_residual_newton"
        mout = write_mirror_output(case_dir / f"mout_{case_name}_residual_newton.nc", result, overwrite=True)
        output = load_mirror_output(mout)
        figure_dir = case_dir / "figures"
        plot_mirror_output(output, outdir=figure_dir, name=f"{case_name}_residual_newton")
        artifacts.append({"case": case_name, "mout": str(mout), "figures": str(figure_dir)})
    return artifacts


def _write_metrics(
    *,
    outdir: Path,
    rows: list[dict[str, object]],
    histories: list[dict[str, object]],
    figures: list[str],
    physical_artifacts: list[dict[str, str]],
) -> Path:
    payload = {
        "rows": rows,
        "histories": histories,
        "figures": figures,
        "physical_artifacts": physical_artifacts,
    }
    path = outdir / "solver_comparison_metrics.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def run_case(
    outdir: Path,
    *,
    cases: tuple[str, ...] = SUPPORTED_CASES,
    maxiter_gd: int = 12,
    maxiter_lbfgs: int = 40,
    maxiter_newton: int = 12,
    line_search_steps: int = 32,
    residual_linear_maxiter: int = 48,
    residual_linear_maxiter_policy: str = "adaptive",
    residual_linear_adaptive_factor: float = 6.0,
    residual_linear_solver: str = "lsmr",
    residual_preconditioner: str = "radial_xi_tridi",
    residual_radial_alpha: float = 0.5,
    residual_lambda_alpha: float = 0.5,
    residual_xi_alpha: float = 0.2,
    gtol: float = 1.0e-12,
    ftol: float = 1.0e-12,
    cylinder_ns: int = 5,
    cylinder_nxi: int = 9,
    cylinder_radius: float = 0.3,
    cylinder_perturbation: float = 0.01,
    two_coil_ns: int = 9,
    two_coil_nxi: int = 17,
    coil_radius: float = 0.35,
    separation: float = 2.0,
    current: float = 1.0e6,
    midplane_radius: float = 0.3,
    two_coil_perturbation: float = 0.02,
    manufactured_ns: int = 5,
    manufactured_nxi: int = 9,
    manufactured_perturbation: float = 0.002,
    write_plots: bool = True,
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    cases = _parse_cases(",".join(cases))
    rows: list[dict[str, object]] = []
    histories: list[dict[str, object]] = []
    physical_problems: list[dict[str, object]] = []
    selected_results: dict[str, object] = {}

    if "cylinder" in cases:
        problem = _cylinder_problem(
            ns=cylinder_ns,
            nxi=cylinder_nxi,
            radius=cylinder_radius,
            perturbation=cylinder_perturbation,
        )
        physical_problems.append(problem)
        case_rows, case_histories, selected = _run_fixed_boundary_case(
            problem,
            maxiter_gd=maxiter_gd,
            maxiter_lbfgs=maxiter_lbfgs,
            maxiter_newton=maxiter_newton,
            line_search_steps=line_search_steps,
            residual_linear_maxiter=residual_linear_maxiter,
            residual_linear_maxiter_policy=residual_linear_maxiter_policy,
            residual_linear_adaptive_factor=residual_linear_adaptive_factor,
            residual_linear_solver=residual_linear_solver,
            residual_preconditioner=residual_preconditioner,
            residual_radial_alpha=residual_radial_alpha,
            residual_lambda_alpha=residual_lambda_alpha,
            residual_xi_alpha=residual_xi_alpha,
            gtol=gtol,
            ftol=ftol,
        )
        rows.extend(case_rows)
        histories.extend(case_histories)
        selected_results["cylinder"] = selected["residual_newton"]

    if "two_coil" in cases:
        problem = _two_coil_problem(
            ns=two_coil_ns,
            nxi=two_coil_nxi,
            coil_radius=coil_radius,
            separation=separation,
            current=current,
            midplane_radius=midplane_radius,
            perturbation=two_coil_perturbation,
        )
        physical_problems.append(problem)
        case_rows, case_histories, selected = _run_fixed_boundary_case(
            problem,
            maxiter_gd=maxiter_gd,
            maxiter_lbfgs=maxiter_lbfgs,
            maxiter_newton=maxiter_newton,
            line_search_steps=line_search_steps,
            residual_linear_maxiter=residual_linear_maxiter,
            residual_linear_maxiter_policy=residual_linear_maxiter_policy,
            residual_linear_adaptive_factor=residual_linear_adaptive_factor,
            residual_linear_solver=residual_linear_solver,
            residual_preconditioner=residual_preconditioner,
            residual_radial_alpha=residual_radial_alpha,
            residual_lambda_alpha=residual_lambda_alpha,
            residual_xi_alpha=residual_xi_alpha,
            gtol=gtol,
            ftol=ftol,
        )
        rows.extend(case_rows)
        histories.extend(case_histories)
        selected_results["two_coil"] = selected["residual_newton"]

    if "manufactured" in cases:
        row, history = _run_manufactured_case(
            ns=manufactured_ns,
            nxi=manufactured_nxi,
            maxiter=maxiter_newton,
            line_search_steps=line_search_steps,
            gtol=gtol,
            ftol=ftol,
            perturbation=manufactured_perturbation,
        )
        rows.append(row)
        histories.append(history)

    figures: list[str] = []
    physical_artifacts: list[dict[str, str]] = []
    if write_plots:
        figures.append(str(_write_residual_history_plot(histories, outdir=outdir)))
        figures.append(str(_write_final_residual_plot(rows, outdir=outdir)))
        boundary_path = _write_boundary_plot(physical_problems, outdir=outdir)
        if boundary_path is not None:
            figures.append(str(boundary_path))
        physical_artifacts = _write_selected_physical_outputs(
            selected_results,
            outdir=outdir,
            write_plots=True,
        )

    return _write_metrics(
        outdir=outdir,
        rows=rows,
        histories=histories,
        figures=figures,
        physical_artifacts=physical_artifacts,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = run_case(
        args.outdir,
        cases=_parse_cases(args.cases),
        maxiter_gd=args.maxiter_gd,
        maxiter_lbfgs=args.maxiter_lbfgs,
        maxiter_newton=args.maxiter_newton,
        line_search_steps=args.line_search_steps,
        residual_linear_maxiter=args.residual_linear_maxiter,
        residual_linear_maxiter_policy=args.residual_linear_maxiter_policy,
        residual_linear_adaptive_factor=args.residual_linear_adaptive_factor,
        residual_linear_solver=args.residual_linear_solver,
        residual_preconditioner=args.residual_preconditioner,
        residual_radial_alpha=args.residual_radial_alpha,
        residual_lambda_alpha=args.residual_lambda_alpha,
        residual_xi_alpha=args.residual_xi_alpha,
        gtol=args.gtol,
        ftol=args.ftol,
        cylinder_ns=args.cylinder_ns,
        cylinder_nxi=args.cylinder_nxi,
        cylinder_radius=args.cylinder_radius,
        cylinder_perturbation=args.cylinder_perturbation,
        two_coil_ns=args.two_coil_ns,
        two_coil_nxi=args.two_coil_nxi,
        coil_radius=args.coil_radius,
        separation=args.separation,
        current=args.current,
        midplane_radius=args.midplane_radius,
        two_coil_perturbation=args.two_coil_perturbation,
        manufactured_ns=args.manufactured_ns,
        manufactured_nxi=args.manufactured_nxi,
        manufactured_perturbation=args.manufactured_perturbation,
        write_plots=not args.no_plots,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
