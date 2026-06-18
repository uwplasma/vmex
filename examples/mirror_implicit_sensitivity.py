"""Validate a tiny reduced-coordinate mirror implicit sensitivity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy import optimize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vmec_jax.mirror import (
    IPrimeProfile,
    MirrorConfig,
    MirrorResolution,
    PressureProfile,
    PsiPrimeProfile,
    axisym_reduced_residual_jacobian_jax,
    axisym_reduced_residual_jax,
    axisym_reduced_residual_linear_solve_jax,
)
from vmec_jax.mirror.core.boundary import MirrorBoundary
from vmec_jax.mirror.core.state import MirrorStateAxisym
from vmec_jax.mirror.solvers.fixed_boundary.optimizers import pack_axisym_reduced_state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("results/mirror/implicit_sensitivity"))
    parser.add_argument("--epsilon", type=float, default=1.0e-5)
    parser.add_argument("--state-ridge", type=float, default=1.0e-3)
    parser.add_argument("--root-tol", type=float, default=1.0e-10)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def _tiny_case():
    config = MirrorConfig(MirrorResolution(ns=5, ntheta=1, nxi=7, mpol=0), z_min=-1.0, z_max=1.0)
    grid = config.build_grid()
    boundary = MirrorBoundary.polynomial_radius(r0=0.3, a2=0.04)
    base = MirrorStateAxisym.from_boundary(grid, boundary)
    s = grid.s_full[:, None]
    xi = grid.xi[None, :]
    state = MirrorStateAxisym(
        a=base.a * (1.0 + 0.01 * s * (1.0 - s) * (1.0 - xi**2)),
        lam=0.005 * s * (xi - np.mean(grid.xi)),
    )
    return grid, boundary, state


def _write_sensitivity_plot(index, implicit, finite_difference, *, outdir: Path) -> Path:
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    error = finite_difference - implicit
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.2), sharex=True)
    axes[0].plot(index, implicit, "-", linewidth=1.4, label="implicit")
    axes[0].plot(index, finite_difference, "--", linewidth=1.2, label="finite difference")
    axes[0].set_ylabel("dx/dp")
    axes[0].legend(fontsize="small")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(index, error, color="tab:red", linewidth=1.2)
    axes[1].set_xlabel("reduced coordinate index")
    axes[1].set_ylabel("difference")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("reduced mirror implicit sensitivity")
    fig.tight_layout()
    path = outdir / "mirror_implicit_sensitivity_components.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def run_case(
    outdir: Path,
    *,
    epsilon: float = 1.0e-5,
    state_ridge: float = 1.0e-3,
    root_tol: float = 1.0e-10,
    write_plots: bool = True,
) -> Path:
    if float(epsilon) <= 0.0:
        raise ValueError("epsilon must be positive")
    if float(state_ridge) <= 0.0:
        raise ValueError("state_ridge must be positive")
    if float(root_tol) <= 0.0:
        raise ValueError("root_tol must be positive")
    outdir.mkdir(parents=True, exist_ok=True)
    grid, boundary, state = _tiny_case()
    psi = PsiPrimeProfile.constant(0.01)
    current = IPrimeProfile.zero()
    pressure = PressureProfile.zero()
    vector = pack_axisym_reduced_state(state, grid, boundary)
    source0 = np.asarray(
        axisym_reduced_residual_jax(
            vector,
            grid,
            boundary,
            psi_prime=psi,
            i_prime=current,
            pressure=pressure,
            mu0=1.0,
        )
    )
    root_residual = np.asarray(
        axisym_reduced_residual_jax(
            vector,
            grid,
            boundary,
            psi_prime=psi,
            i_prime=current,
            pressure=pressure,
            source_vector=source0,
            state_ridge=state_ridge,
            reference_vector=vector,
            mu0=1.0,
        )
    )
    source_direction = np.sin(np.linspace(0.1, 1.3, vector.size))
    implicit_sensitivity = np.asarray(
        axisym_reduced_residual_linear_solve_jax(
            vector,
            source_direction,
            grid,
            boundary,
            psi_prime=psi,
            i_prime=current,
            pressure=pressure,
            source_vector=source0,
            state_ridge=state_ridge,
            reference_vector=vector,
            mu0=1.0,
        )
    )
    source_eps = source0 + float(epsilon) * source_direction

    def residual(items):
        return np.asarray(
            axisym_reduced_residual_jax(
                items,
                grid,
                boundary,
                psi_prime=psi,
                i_prime=current,
                pressure=pressure,
                source_vector=source_eps,
                state_ridge=state_ridge,
                reference_vector=vector,
                mu0=1.0,
            )
        )

    def jacobian(items):
        return np.asarray(
            axisym_reduced_residual_jacobian_jax(
                items,
                grid,
                boundary,
                psi_prime=psi,
                i_prime=current,
                pressure=pressure,
                source_vector=source_eps,
                state_ridge=state_ridge,
                reference_vector=vector,
                mu0=1.0,
            )
        )

    solved = optimize.root(
        residual,
        vector + float(epsilon) * implicit_sensitivity,
        jac=jacobian,
        method="hybr",
        options={"xtol": min(1.0e-11, float(root_tol)), "maxfev": 120},
    )
    perturbed_residual_norm = float(np.linalg.norm(residual(solved.x)))
    finite_difference_sensitivity = (solved.x - vector) / float(epsilon)
    difference = finite_difference_sensitivity - implicit_sensitivity
    relative_error = float(np.linalg.norm(difference) / max(np.linalg.norm(implicit_sensitivity), np.finfo(float).tiny))
    max_abs_error = float(np.max(np.abs(difference)))
    root_accepted = bool(perturbed_residual_norm < float(root_tol))
    accepted = bool(root_accepted and relative_error < 1.0e-3)
    figures: dict[str, str] = {}
    if write_plots:
        figures["components"] = str(
            _write_sensitivity_plot(
                np.arange(vector.size),
                implicit_sensitivity,
                finite_difference_sensitivity,
                outdir=outdir / "figures",
            )
        )

    metrics = {
        "vector_size": int(vector.size),
        "epsilon": float(epsilon),
        "state_ridge": float(state_ridge),
        "root_residual_norm": float(np.linalg.norm(root_residual)),
        "perturbed_root_solver_success": bool(solved.success),
        "perturbed_root_success": root_accepted,
        "perturbed_root_message": str(solved.message),
        "perturbed_residual_norm": perturbed_residual_norm,
        "relative_error": relative_error,
        "max_abs_error": max_abs_error,
        "accepted": accepted,
        "figures": figures,
    }
    path = outdir / "mirror_implicit_sensitivity_metrics.json"
    path.write_text(json.dumps(metrics, indent=2) + "\n")
    return path


def main() -> None:
    args = build_parser().parse_args()
    path = run_case(
        args.outdir,
        epsilon=args.epsilon,
        state_ridge=args.state_ridge,
        root_tol=args.root_tol,
        write_plots=not args.no_plots,
    )
    print(path)


if __name__ == "__main__":
    main()
