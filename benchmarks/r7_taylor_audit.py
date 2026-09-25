"""R7 frozen-state Taylor, representability, and representation-floor audit.

Read-only diagnostic for PR #448.  At one saved native checkpoint it measures:

1. the stationarity change caused by one-ulp perturbations of the physical
   native coefficients (the float64 representation floor of ``P g``);
2. the exact-Hessian Newton direction and the Taylor remainders
   ``T_g(h) = P[g(c+hp) - g(c) - h H p]`` on a logarithmic ladder;
3. the same ladder for a random feasible direction of equal norm;
4. how many physical coefficients change after floating-point addition and
   the realized versus intended physical increment.

It solves no equilibrium and writes no checkpoint.  Run under run_checked.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from scipy import sparse  # noqa: E402
from scipy.sparse.linalg import splu  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import polish_recovery_sparse as prs  # noqa: E402

FIELDS = ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ulp-trials", type=int, default=4)
    parser.add_argument("--stable-derivatives", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    if args.stable_derivatives:
        import functools

        prs.make_variational_plan = functools.partial(prs.make_variational_plan, stable_derivatives=True)
    if _sha(args.state) != args.expected_sha256:
        raise ValueError("checkpoint hash mismatch")
    report: dict = {
        "schema": "vmex-r7-taylor-audit/1",
        "state": str(args.state),
        "state_sha256": args.expected_sha256,
        "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        "generator_sha256": _sha(Path(__file__)),
        "environment": {"jax": jax.__version__, "numpy": np.__version__, "precision": "float64", "device": str(jax.devices()[0])},
        "production_qualified": False,
        "stable_derivatives": bool(args.stable_derivatives),
    }
    base, accepted, plan, layout, gauge, scale, coords, _ = prs._load_original_problem(args.state)
    force, constraint_function = prs._linear_problem(base, plan, layout, gauge, scale, coords)
    C = prs.native_tangential_gauge_matrix(base, layout, gauge, scale).tocsr()
    n, q = C.shape[1], C.shape[0]
    projector = splu(sparse.bmat([[sparse.eye(n, format="csc"), C.T], [C, None]], format="csc"))

    def project(v):
        return projector.solve(np.r_[np.asarray(v, dtype=float), np.zeros(q)])[:n]

    def gradient_for(state_base, c):
        def objective(value):
            r = prs.native_physical_force_residual(
                value, state_base, layout, gauge, scale, prs.FORCE_SCALE, prs.VOLUME_SCALE
            )
            return 0.5 * jnp.vdot(r, r)

        return jax.jit(jax.grad(objective))(c)

    gradient_function = jax.jit(jax.grad(lambda c: 0.5 * jnp.vdot(force(c), force(c))))
    residual = np.asarray(force(coords))
    g0 = np.asarray(gradient_function(coords))
    pg0 = project(g0)
    normal, _, local = prs._local_normal_system(force, coords, residual, base, plan, layout, gauge, scale)
    a_frob = float(np.sqrt(normal.diagonal().sum()))
    eta_scale = a_frob * float(np.linalg.norm(residual))
    report["current"] = {
        "residual_norm": float(np.linalg.norm(residual)),
        "full_gradient_norm": float(np.linalg.norm(g0)),
        "projected_gradient_norm": float(np.linalg.norm(pg0)),
        "operator_frobenius": a_frob,
        "eta": float(np.linalg.norm(pg0)) / eta_scale,
        "eta_gate_projected_gradient_threshold": 1.0e-8 * eta_scale,
        "local_normal_assembly_seconds": local["assembly_seconds"],
    }

    # 1. Representation floor: perturb the stored physical coefficients by one
    # ulp on a random half of the entries, holding c fixed.  In exact arithmetic
    # the state is unchanged to 1e-16 relative, so the change in P g is the
    # float64 floor of the evaluated stationarity map at this state.
    rng = np.random.default_rng(7448)
    ulp_rows = []
    for trial in range(args.ulp_trials):
        changes = {}
        for name in FIELDS:
            value = np.asarray(getattr(base, name))
            mask = rng.random(value.shape) < 0.5
            direction = np.where(rng.random(value.shape) < 0.5, -np.inf, np.inf)
            changes[name] = jnp.asarray(np.where(mask, np.nextafter(value, direction), value))
        perturbed = replace(base, **changes)
        g1 = np.asarray(gradient_for(perturbed, coords))
        dp = project(g1 - g0)
        ulp_rows.append(
            {
                "trial": trial,
                "full_gradient_change": float(np.linalg.norm(g1 - g0)),
                "projected_gradient_change": float(np.linalg.norm(dp)),
                "projected_change_over_gate_threshold": float(np.linalg.norm(dp)) / (1.0e-8 * eta_scale),
                "eta_after": float(np.linalg.norm(project(g1))) / eta_scale,
            }
        )
    report["one_ulp_base_perturbation"] = ulp_rows
    # Same measurement with the ulp perturbation applied only to lambda.
    json_path = args.output
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2))

    # 2. Newton direction and Taylor ladders.
    defect = np.asarray(constraint_function(coords))
    step, linear, _, _, hessian_action, _ = prs._exact_stationarity_step(force, coords, C, defect, normal)
    report["newton_linear_certificate"] = {k: v for k, v in linear.items() if not isinstance(v, dict)}
    hp = np.asarray(hessian_action(jnp.asarray(step)))
    report["newton_model"] = {
        "projected_linear_model_norm": float(np.linalg.norm(project(g0 + hp))),
        "pHp": float(step @ hp),
        "gTp": float(g0 @ step),
        "predicted_objective_change": float(g0 @ step + 0.5 * step @ hp),
    }
    physical0 = {name: np.asarray(getattr(accepted, name)) for name in FIELDS}
    v = project(rng.standard_normal(n))
    v *= np.linalg.norm(step) / np.linalg.norm(v)
    hv = np.asarray(hessian_action(jnp.asarray(v)))
    ladders = {}
    for label, direction, h_direction in (("newton", step, hp), ("random_feasible", v, hv)):
        rows = []
        for h in (4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 1e-1, 1e-2, 1e-3, 1e-4):
            candidate = coords + h * jnp.asarray(direction)
            state = prs.apply_high_order_correction(base, layout.unpack(scale * candidate))
            intended = layout.unpack(scale * jnp.asarray(h * direction))
            changed = 0
            total = 0
            realized_error = 0.0
            intended_norm = 0.0
            for name in FIELDS:
                new = np.asarray(getattr(state, name))
                delta = new - physical0[name]
                target = np.asarray(getattr(intended, name))
                changed += int(np.count_nonzero(delta))
                total += int(np.count_nonzero(target))
                realized_error += float(np.sum((delta - target) ** 2))
                intended_norm += float(np.sum(target**2))
            r_h = np.asarray(force(candidate))
            g_h = np.asarray(gradient_function(candidate))
            dr = r_h - residual
            rows.append(
                {
                    "h": h,
                    "eta": float(np.linalg.norm(project(g_h))) / (a_frob * float(np.linalg.norm(r_h))),
                    "projected_gradient_norm": float(np.linalg.norm(project(g_h))),
                    "T_g_norm": float(np.linalg.norm(project(g_h - g0 - h * h_direction))),
                    "linear_term_norm": float(np.linalg.norm(project(h * h_direction))),
                    "objective_change_from_increments": float(residual @ dr + 0.5 * dr @ dr),
                    "objective_model_change": float(h * (g0 @ direction) + 0.5 * h * h * (direction @ h_direction)),
                    "physical_coefficients_changed": changed,
                    "physical_coefficients_targeted": total,
                    "realized_increment_relative_error": float(np.sqrt(realized_error / max(intended_norm, 1e-300))),
                    "gauge_norm": float(np.linalg.norm(np.asarray(constraint_function(candidate)))),
                }
            )
        ladders[label] = rows
    report["taylor_ladders"] = ladders
    report["elapsed_seconds"] = time.perf_counter() - started
    report["complete"] = True
    json_path.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
