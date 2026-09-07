"""E2 fixed-profile dense reference: full R/Z versus the structured subspace.

Uses the actual constrained layout from E1. Both runs start from the same
shaped-tokamak seed with frozen flux and pressure profiles (GAMMA=0). This
is a bounded diagnostic, not a production solver or a fixed-current solve.
Augmented least squares uses SciPy's independent GELSD/GELSY implementations:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lstsq.html
"""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform
import resource
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import lstsq

import vmex
from benchmarks._provenance import assert_repo_vmex, git_state
from benchmarks.e1_functional_consistency import shaped_state
from vmex.core.polish import _physical_equation_basis, apply_high_order_correction
from vmex.core.profiles import MU0
from vmex.core.radial_basis import _span_quadrature
from vmex.core.strong_force import _basic_fields, _point_force, certify_strong_force


def certificate(state):
    report = certify_strong_force(state, angular_multiplier=4, radial_order_increment=8)
    return {name: float(getattr(report, name)) for name in (
        "absolute_l2", "absolute_linf", "normalized_l2", "near_axis_l2",
        "bulk_l2", "edge_l2", "minimum_signed_jacobian", "radial_refinement_difference")}


def spectrum(matrix):
    singular = np.linalg.svd(matrix, compute_uv=False)
    return {"shape": list(matrix.shape), "bytes": matrix.nbytes,
            "rank_rtol_1e-10": int(np.sum(singular > singular[0] * 1e-10)),
            "singular_values": singular.tolist()}


def experiment(native, layout, steps, order, angles, damping):
    rho, weights = _span_quadrature(np.sqrt(native.radial_basis.breakpoints), order)
    points = jnp.asarray([(r, theta, 0.0) for r in rho
                          for theta in np.arange(angles) * 2 * np.pi / angles])
    weights = jnp.asarray(np.repeat(weights, angles) * (2 * np.pi)**2 / angles)
    zero = jnp.zeros(layout.size)

    def state(vector):
        return apply_high_order_correction(native, layout.unpack(vector))

    @jax.jit
    def fields(vector):
        jacobian, _, _, force, *_ = jax.vmap(lambda point: _point_force(state(vector), point))(points)
        return jacobian, force

    jacobian, _ = fields(zero)
    # Freeze the reference volume measure. Scaling is explicit column-L2
    # equilibration below, not a claim to reproduce production Ruiz scaling.
    volume_weights = weights * jnp.abs(jacobian)
    measure = jnp.sqrt(volume_weights / jnp.sum(volume_weights))
    residual = jax.jit(lambda vector: (measure[:, None] * fields(vector)[1]).reshape(-1))
    jac = jax.jit(jax.jacfwd(residual))

    def energy(vector):
        def density(point):
            _, g, _, _, B, p = _basic_fields(state(vector), point)
            return (jnp.vdot(B, B) / (2 * MU0) - p) * jnp.abs(g)
        return weights @ jax.vmap(density)(points)

    estimated_bytes = (len(points) * 3 * layout.size + layout.size**2) * 8
    if estimated_bytes > 2 * 1024**3:
        raise MemoryError("E2 reference matrices exceed the 2 GiB plan budget")
    started = time.perf_counter()
    J0 = np.asarray(jac(zero))
    H = np.asarray(jax.jit(jax.hessian(energy))(zero))
    build_seconds = time.perf_counter() - started
    direction = jnp.cos(jnp.arange(layout.size)) / np.sqrt(layout.size)
    finite_difference = np.asarray((residual(1e-6 * direction) - residual(-1e-6 * direction)) / 2e-6)
    jv = J0 @ np.asarray(direction)
    initial = certificate(native)
    results = {"initial_certificate": initial, "reference_build_seconds": build_seconds,
               "reference_matrix_bytes": estimated_bytes,
               "jacobian_fd_relative": float(np.linalg.norm(jv - finite_difference) / np.linalg.norm(jv)),
               "full_jacobian": spectrum(J0), "full_energy_hessian": {
                   "symmetry_relative": float(np.linalg.norm(H - H.T) / np.linalg.norm(H)),
                   "eigenvalues": np.linalg.eigvalsh((H + H.T) / 2).tolist()}, "charts": {}}
    for name, basis in (("full_RZ", np.eye(layout.size)),
                        ("structured", _physical_equation_basis(layout))):
        vector = np.zeros(layout.size)
        history = []
        for iteration in range(steps):
            r = np.asarray(residual(jnp.asarray(vector)))
            J = (J0 if iteration == 0 else np.asarray(jac(jnp.asarray(vector)))) @ basis
            column_norm = np.linalg.norm(J, axis=0)
            scale = 1 / np.maximum(column_norm, max(float(column_norm.max()) * 1e-12, 1e-30))
            scaled = J * scale
            augmented = np.vstack((scaled, np.sqrt(damping) * np.eye(basis.shape[1])))
            rhs = np.concatenate((-r, np.zeros(basis.shape[1])))
            started = time.perf_counter()
            y, _, rank, _ = lstsq(augmented, rhs, lapack_driver="gelsd")
            qr, _, _, _ = lstsq(augmented, rhs, lapack_driver="gelsy")
            solve_seconds = time.perf_counter() - started
            step = basis @ (scale * y)
            cost = float(np.vdot(r, r) / 2)
            trials = []
            accepted = False
            for fraction in (1.0, 0.5, 0.25, 0.125, 0.0625):
                candidate = jnp.asarray(vector + fraction * step)
                g, force = fields(candidate)
                valid = bool(jnp.all(jnp.isfinite(force)) & jnp.all(native.jacobian_sign * g > 0))
                trial_cost = float(jnp.vdot(residual(candidate), residual(candidate)) / 2) if valid else None
                trials.append({"fraction": fraction, "admissible_on_solve_grid": valid, "cost": trial_cost})
                if valid and trial_cost < cost:
                    vector = np.asarray(candidate)
                    accepted = True
                    break
            equation_defect = augmented.T @ (augmented @ y - rhs)
            record = {"iteration": iteration, "cost": cost, "accepted": accepted,
                      "column_scale": scale.tolist(), "scaled_jacobian": spectrum(scaled),
                      "augmented_rank": int(rank), "qr_svd_step_relative": float(
                          np.linalg.norm(qr - y) / max(np.linalg.norm(y), 1e-30)),
                      "augmented_optimality_relative": float(np.linalg.norm(equation_defect) /
                          max(np.linalg.norm(augmented.T @ rhs), 1e-30)),
                      "true_linear_residual_norm": float(np.linalg.norm(J @ (scale * y) + r)),
                      "predicted_full_step_cost": float(np.linalg.norm(J @ (scale * y) + r)**2 / 2),
                      "factorizations_seconds": solve_seconds, "trials": trials}
            history.append(record)
            print(json.dumps({"chart": name, "iteration": iteration, "cost": cost,
                              "accepted": accepted, "last_trial": trials[-1]}), flush=True)
            if not accepted:
                break
        final = certificate(state(jnp.asarray(vector)))
        results["charts"][name] = {"coordinates": basis.shape[1], "history": history,
                                  "final_certificate": final,
                                  "absolute_l2_improvement": initial["absolute_l2"] / final["absolute_l2"],
                                  "final_coordinates": vector.tolist()}
    results["verification_passed"] = (
        results["jacobian_fd_relative"] < 1e-6
        and results["full_energy_hessian"]["symmetry_relative"] < 1e-10
        and all(np.isfinite(list(chart["final_certificate"].values())).all()
                and chart["final_certificate"]["minimum_signed_jacobian"] > 0
                and all(record["qr_svd_step_relative"] < 1e-9
                        and record["augmented_optimality_relative"] < 1e-9
                        for record in chart["history"])
                for chart in results["charts"].values()))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--order", type=int, default=6)
    parser.add_argument("--angles", type=int, default=24)
    parser.add_argument("--damping", type=float, default=1e-3)
    args = parser.parse_args()
    if not jax.config.x64_enabled or not 1 <= args.steps <= 5 or not 4 <= args.order <= 12 or not 16 <= args.angles <= 64:
        parser.error("require float64, steps 1..5, order 4..12, angles 16..64")
    if not np.isfinite(args.damping) or args.damping <= 0:
        parser.error("damping must be finite and positive")
    def expired(*_):
        raise TimeoutError("E2 600 s diagnostic budget")
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(600)
    provenance = git_state(ROOT)
    provenance.update(vmex_module=assert_repo_vmex(vmex.__file__, ROOT),
                      python=platform.python_version(), platform=platform.system(),
                      versions={name: version(name) for name in ("jax", "jaxlib", "scipy", "numpy", "solvax")},
                      device=str(jax.devices()[0]), precision="float64")
    started = time.perf_counter()
    output = {"_provenance": provenance, "case": "shaped tokamak, frozen iota/pressure, GAMMA=0",
              "settings": {k: v for k, v in vars(args).items() if k != "output"}}
    try:
        output["result"] = experiment(*shaped_state(), args.steps, args.order, args.angles, args.damping)
        output["status"] = "completed diagnostic; QA and nonlinear convergence not established"
        if not output["result"]["verification_passed"]:
            raise RuntimeError("E2 reference consistency or geometry check failed")
    except Exception as error:
        output["status"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        output["seconds"] = time.perf_counter() - started
        output["peak_rss_MiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
            1024**2 if platform.system() == "Darwin" else 1024)
        args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
