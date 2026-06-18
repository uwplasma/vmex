"""Diagnose fixed-boundary mirror L-BFGS convergence on a two-coil boundary."""

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("results/mirror/fixed_boundary_solve_diagnostic"))
    parser.add_argument("--ns-array", type=str, default="31")
    parser.add_argument("--nxi", type=int, default=33)
    parser.add_argument("--maxiter", type=int, default=2000)
    parser.add_argument("--line-search-steps", type=int, default=64)
    parser.add_argument("--optimizer", type=str, default="lbfgs", choices=("lbfgs", "residual_newton"))
    parser.add_argument("--residual-linear-maxiter", type=int, default=16)
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
        choices=("lsmr", "lsqr", "dense_lstsq", "block_dense_lstsq", "block_lsmr"),
        help="Linear solver for residual-Newton corrections.",
    )
    parser.add_argument(
        "--residual-compare-dense-step",
        action="store_true",
        help="On small matrix-free runs, compare each Krylov correction with the dense reduced-Hessian step.",
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
    parser.add_argument("--coil-radius", type=float, default=0.35)
    parser.add_argument("--separation", type=float, default=2.0)
    parser.add_argument("--current", type=float, default=1.0e6)
    parser.add_argument("--midplane-radius", type=float, default=0.3)
    parser.add_argument("--perturbation", type=float, default=0.02)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def _parse_int_array(value: str) -> tuple[int, ...]:
    items = [item for item in re.split(r"[\s,]+", str(value).strip()) if item]
    if not items:
        raise ValueError("ns-array must contain at least one integer")
    values = tuple(int(item) for item in items)
    if any(item < 3 for item in values):
        raise ValueError("all ns values must be at least 3")
    return values


def _perturbed_initial_state(config: MirrorConfig, boundary, *, amplitude: float) -> MirrorStateAxisym:
    grid = config.build_grid()
    base = MirrorStateAxisym.from_boundary(grid, boundary)
    s = grid.s_full[:, None]
    xi = grid.xi[None, :]
    shape = s * (1.0 - s) * (1.0 - xi**2)
    a = base.a * (1.0 + float(amplitude) * shape)
    return MirrorStateAxisym(a=a, lam=np.zeros_like(a))


def _run_one(
    outdir: Path,
    *,
    ns: int,
    nxi: int,
    maxiter: int,
    line_search_steps: int,
    optimizer: str,
    residual_linear_maxiter: int,
    residual_linear_maxiter_policy: str,
    residual_linear_adaptive_factor: float,
    residual_linear_solver: str,
    residual_compare_dense_step: bool,
    residual_preconditioner: str,
    residual_radial_alpha: float,
    residual_lambda_alpha: float,
    residual_xi_alpha: float,
    gtol: float,
    ftol: float,
    coil_radius: float,
    separation: float,
    current: float,
    midplane_radius: float,
    perturbation: float,
    write_plots: bool,
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
    initial_state = _perturbed_initial_state(config, boundary, amplitude=perturbation)
    result = run_mirror_fixed_boundary(
        config,
        boundary,
        psi_prime=PsiPrimeProfile.constant(psi_value),
        i_prime=IPrimeProfile.zero(),
        pressure=PressureProfile.zero(),
        initial_state=initial_state,
        options=MirrorSolveOptions(
            optimizer=optimizer,
            maxiter=maxiter,
            tolerance=gtol,
            ftol=ftol,
            line_search_steps=line_search_steps,
            residual_linear_maxiter=residual_linear_maxiter,
            residual_linear_maxiter_policy=residual_linear_maxiter_policy,
            residual_linear_adaptive_factor=residual_linear_adaptive_factor,
            residual_linear_solver=residual_linear_solver,
            residual_compare_dense_step=residual_compare_dense_step,
            residual_preconditioner=residual_preconditioner,
            residual_radial_alpha=residual_radial_alpha,
            residual_lambda_alpha=residual_lambda_alpha,
            residual_xi_alpha=residual_xi_alpha,
            mu0=1.0,
        ),
    )
    case_dir = outdir / f"ns{int(ns)}_nxi{int(nxi)}"
    mout = write_mirror_output(
        case_dir / f"mout_fixed_boundary_solve_ns{int(ns)}_nxi{int(nxi)}.nc", result, overwrite=True
    )
    output = load_mirror_output(mout)
    if write_plots:
        plot_mirror_output(output, outdir=case_dir / "figures", name=f"fixed_boundary_solve_ns{int(ns)}_nxi{int(nxi)}")

    summary = result.optimizer_summaries[-1] if result.optimizer_summaries else None
    first = result.trace[0]
    final = result.final_trace
    return {
        "ns": int(ns),
        "nxi": int(nxi),
        "maxiter": int(maxiter),
        "line_search_steps": int(line_search_steps),
        "optimizer": str(optimizer),
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
        "residual_linear_istop_last": None if summary is None else summary.residual_linear_istop_last,
        "residual_linear_iterations_last": None if summary is None else summary.residual_linear_iterations_last,
        "residual_linear_iterations_total": None if summary is None else summary.residual_linear_iterations_total,
        "residual_linear_residual_norm_last": None if summary is None else summary.residual_linear_residual_norm_last,
        "residual_linear_normal_residual_norm_last": None
        if summary is None
        else summary.residual_linear_normal_residual_norm_last,
        "residual_linear_condition_estimate_last": None
        if summary is None
        else summary.residual_linear_condition_estimate_last,
        "residual_compare_dense_step": bool(residual_compare_dense_step),
        "residual_dense_step_norm_last": None if summary is None else summary.residual_dense_step_norm_last,
        "residual_dense_step_cosine_last": None if summary is None else summary.residual_dense_step_cosine_last,
        "residual_dense_step_relative_error_last": None
        if summary is None
        else summary.residual_dense_step_relative_error_last,
        "residual_preconditioner": str(residual_preconditioner),
        "residual_radial_alpha": float(residual_radial_alpha),
        "residual_lambda_alpha": float(residual_lambda_alpha),
        "residual_xi_alpha": float(residual_xi_alpha),
        "gtol": float(gtol),
        "ftol": float(ftol),
        "mout": str(mout),
        "trace_steps": int(len(result.trace)),
        "optimizer_nit": int(summary.nit) if summary is not None else 0,
        "optimizer_nfev": int(summary.nfev) if summary is not None else 0,
        "optimizer_njev": int(summary.njev) if summary is not None else 0,
        "optimizer_success": bool(summary.success) if summary is not None else False,
        "optimizer_accepted": bool(summary.accepted) if summary is not None else False,
        "optimizer_status": int(summary.status) if summary is not None else -1,
        "optimizer_message": str(summary.message) if summary is not None else "",
        "optimizer_rejection_reason": str(summary.rejection_reason) if summary is not None else "",
        "optimizer_candidate_energy_total": None if summary is None else summary.candidate_energy_total,
        "optimizer_candidate_residual_norm": None if summary is None else summary.candidate_residual_norm,
        "optimizer_candidate_min_a": None if summary is None else summary.candidate_min_a,
        "optimizer_candidate_min_sqrtg": None if summary is None else summary.candidate_min_sqrtg,
        "optimizer_candidate_energy_improved": None if summary is None else summary.candidate_energy_improved,
        "optimizer_candidate_positive_radius": None if summary is None else summary.candidate_positive_radius,
        "optimizer_candidate_positive_jacobian": None if summary is None else summary.candidate_positive_jacobian,
        "initial_energy_total": float(first.energy_total),
        "final_energy_total": float(final.energy_total),
        "energy_drop": float(first.energy_total - final.energy_total),
        "initial_residual_norm": float(first.residual_norm),
        "final_residual_norm": float(final.residual_norm),
        "residual_drop": float(first.residual_norm - final.residual_norm),
        "final_fsq": float(final.fsq),
        "final_normalized_force": float(final.normalized_force),
        "reached_projected_gtol": bool(final.residual_norm <= float(gtol)),
        "min_sqrtg": float(final.min_sqrtg),
        "mirror_ratio": float(final.mirror_ratio),
    }


def run_case(
    outdir: Path,
    *,
    ns_array: tuple[int, ...] = (31,),
    nxi: int = 33,
    maxiter: int = 2000,
    line_search_steps: int = 64,
    optimizer: str = "lbfgs",
    residual_linear_maxiter: int = 16,
    residual_linear_maxiter_policy: str = "adaptive",
    residual_linear_adaptive_factor: float = 6.0,
    residual_linear_solver: str = "lsmr",
    residual_compare_dense_step: bool = False,
    residual_preconditioner: str = "radial_xi_tridi",
    residual_radial_alpha: float = 0.5,
    residual_lambda_alpha: float = 0.5,
    residual_xi_alpha: float = 0.2,
    gtol: float = 1.0e-12,
    ftol: float = 1.0e-12,
    coil_radius: float = 0.35,
    separation: float = 2.0,
    current: float = 1.0e6,
    midplane_radius: float = 0.3,
    perturbation: float = 0.02,
    write_plots: bool = True,
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    rows = [
        _run_one(
            outdir,
            ns=ns,
            nxi=nxi,
            maxiter=maxiter,
            line_search_steps=line_search_steps,
            optimizer=optimizer,
            residual_linear_maxiter=residual_linear_maxiter,
            residual_linear_maxiter_policy=residual_linear_maxiter_policy,
            residual_linear_adaptive_factor=residual_linear_adaptive_factor,
            residual_linear_solver=residual_linear_solver,
            residual_compare_dense_step=residual_compare_dense_step,
            residual_preconditioner=residual_preconditioner,
            residual_radial_alpha=residual_radial_alpha,
            residual_lambda_alpha=residual_lambda_alpha,
            residual_xi_alpha=residual_xi_alpha,
            gtol=gtol,
            ftol=ftol,
            coil_radius=coil_radius,
            separation=separation,
            current=current,
            midplane_radius=midplane_radius,
            perturbation=perturbation,
            write_plots=write_plots,
        )
        for ns in ns_array
    ]
    path = outdir / "fixed_boundary_solve_diagnostic.json"
    path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = run_case(
        args.outdir,
        ns_array=_parse_int_array(args.ns_array),
        nxi=args.nxi,
        maxiter=args.maxiter,
        line_search_steps=args.line_search_steps,
        optimizer=args.optimizer,
        residual_linear_maxiter=args.residual_linear_maxiter,
        residual_linear_maxiter_policy=args.residual_linear_maxiter_policy,
        residual_linear_adaptive_factor=args.residual_linear_adaptive_factor,
        residual_linear_solver=args.residual_linear_solver,
        residual_compare_dense_step=args.residual_compare_dense_step,
        residual_preconditioner=args.residual_preconditioner,
        residual_radial_alpha=args.residual_radial_alpha,
        residual_lambda_alpha=args.residual_lambda_alpha,
        residual_xi_alpha=args.residual_xi_alpha,
        gtol=args.gtol,
        ftol=args.ftol,
        coil_radius=args.coil_radius,
        separation=args.separation,
        current=args.current,
        midplane_radius=args.midplane_radius,
        perturbation=args.perturbation,
        write_plots=not args.no_plots,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
