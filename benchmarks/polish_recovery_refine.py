"""Exactly refine the recovered P3 B checkpoint and take bounded feasible steps."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import jax
import jax.numpy as jnp
import numpy as np

from polish_recovery_p3 import (
    _checkpoint_arrays,
    _constraint_qr,
    _dense_reference_memory,
    _final_projected_gradient,
    _feasible_least_squares_step,
    _write_npz_atomic,
)
from vmex.core.polish import apply_high_order_correction, make_native_correction_layout
from vmex.core.polish_variational import (
    evaluate_tensorized_strong_force,
    make_native_gauge_plan,
    make_variational_plan,
    minimum_signed_jacobian,
    native_coordinate_scales,
    native_physical_force_residual,
    native_tangential_gauge_residual,
)
from vmex.core.radial_basis import BSplineBasis
from vmex.core.strong_force import (
    HighOrderEquilibriumState,
    certify_strong_force,
    evaluate_high_order_fields,
    evaluate_strong_force,
    insert_high_order_state_knots,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_state(path: Path, prefix: str = "accepted") -> HighOrderEquilibriumState:
    with np.load(path, allow_pickle=False) as data:
        schema = str(np.asarray(data["schema"]).item())
        if schema not in {
            "vmex.polish-recovery-native-state/1",
            "vmex.polish-recovery-native-state/2",
        }:
            raise ValueError(f"unsupported native checkpoint schema {schema!r}")
        degree = int(np.asarray(data["basis_degree"]).item())
        if bool(np.asarray(data["basis_periodic"]).item()):
            raise ValueError("recovery checkpoint requires an open radial basis")
        basis = BSplineBasis.clamped(
            np.asarray(data["basis_breakpoints"]), degree=degree
        )
        stored_knots = np.asarray(data["basis_knots"], dtype=float)
        if stored_knots.shape != basis.knots.shape or not np.array_equal(
            stored_knots, basis.knots
        ):
            raise ValueError(
                "checkpoint knot vector is not representable by the current "
                "simple-interior clamped basis"
            )
        m = np.asarray(data["m"])
        n = np.asarray(data["n"])
        if (
            m.ndim != 1
            or n.shape != m.shape
            or not np.issubdtype(m.dtype, np.integer)
            or not np.issubdtype(n.dtype, np.integer)
            or np.unique(np.stack((m, n), axis=1), axis=0).shape[0] != m.size
        ):
            raise ValueError("checkpoint mode table must contain unique integer pairs")
        values = {
            name: jnp.asarray(data[f"{prefix}_{name}"])
            for name in (
                "R_cos",
                "R_sin",
                "Z_cos",
                "Z_sin",
                "L_cos",
                "L_sin",
                "phipf",
                "chipf",
                "pressure",
                "boundary_R_cos",
                "boundary_R_sin",
                "boundary_Z_cos",
                "boundary_Z_sin",
            )
        }
        return HighOrderEquilibriumState(
            radial_basis=basis,
            m=np.asarray(m, dtype=int),
            n=np.asarray(n, dtype=int),
            nfp=int(np.asarray(data["nfp"]).item()),
            jacobian_sign=int(np.asarray(data["jacobian_sign"]).item()),
            **values,
        )


def _relative_error(actual, expected) -> float:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    return float(
        np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1.0)
    )


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-state",
        type=Path,
        default=Path("benchmarks/polish_recovery_r1_p3_state.npz"),
    )
    parser.add_argument(
        "--output-state",
        type=Path,
        default=Path("benchmarks/polish_recovery_r3_refined_state.npz"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("benchmarks/polish_recovery_r3_refinement.json"),
    )
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument(
        "--adaptive-count",
        type=int,
        default=0,
        help="insert midpoints of this many largest force-contribution spans; 0 bisects all",
    )
    parser.add_argument("--dense-memory-gib", type=float, default=2.0)
    args = parser.parse_args()
    if args.max_steps < 0 or args.adaptive_count < 0 or args.dense_memory_gib <= 0.0:
        parser.error("step/count values must be nonnegative and memory cap positive")

    started = time.perf_counter()
    force_scale = 5915447.712414409
    volume_scale = 633.7993467060758
    coarse = _load_state(args.input_state)
    old_breakpoints = np.asarray(coarse.radial_basis.breakpoints)
    span_scores = None
    selected_spans = np.arange(old_breakpoints.size - 1)
    if args.adaptive_count:
        span_count = old_breakpoints.size - 1
        if args.adaptive_count >= span_count:
            parser.error("--adaptive-count must be smaller than the current span count")
        indicator_order = 4
        indicator_plan = make_variational_plan(coarse, radial_order=indicator_order)
        indicator = evaluate_tensorized_strong_force(coarse, indicator_plan)
        weighted_force = (
            indicator_plan.quadrature_weights
            * coarse.jacobian_sign
            * indicator.sqrt_g
            * jnp.sum(indicator.force**2, axis=-1)
        )
        radial_score = np.asarray(jnp.sum(weighted_force, axis=(1, 2)))
        span_scores = radial_score.reshape(span_count, indicator_order).sum(axis=1)
        selected_spans = np.sort(
            np.argsort(span_scores)[-args.adaptive_count :]
        )
    inserted_knots = 0.5 * (
        old_breakpoints[selected_spans] + old_breakpoints[selected_spans + 1]
    )
    transfer_started = time.perf_counter()
    refined = insert_high_order_state_knots(coarse, inserted_knots)
    transfer_seconds = time.perf_counter() - transfer_started

    rho = jnp.asarray(np.linspace(0.00017, 0.99983, 31))
    theta = jnp.asarray(np.mod(np.arange(31) * 2.399963229728653, 2.0 * np.pi))
    zeta = jnp.zeros_like(rho)
    coarse_fields = evaluate_high_order_fields(coarse, rho, theta, zeta)
    refined_fields = evaluate_high_order_fields(refined, rho, theta, zeta)
    coarse_force = evaluate_strong_force(coarse, rho, theta, zeta)
    refined_force = evaluate_strong_force(refined, rho, theta, zeta)
    s_points = np.linspace(0.00017, 0.99983, 301)
    spline_transfer_errors = {}
    for derivative in range(3):
        errors = []
        for name in (
            "R_cos",
            "R_sin",
            "Z_cos",
            "Z_sin",
            "L_cos",
            "L_sin",
            "phipf",
            "chipf",
            "pressure",
        ):
            errors.append(
                _relative_error(
                    refined.radial_basis.evaluate(
                        getattr(refined, name), s_points, derivative=derivative
                    ),
                    coarse.radial_basis.evaluate(
                        getattr(coarse, name), s_points, derivative=derivative
                    ),
                )
            )
        spline_transfer_errors[f"derivative_{derivative}"] = max(errors)
    transfer_errors = {
        name: _relative_error(getattr(refined_fields, name), getattr(coarse_fields, name))
        for name in (
            "position",
            "dposition_drho",
            "dposition_dtheta",
            "dposition_dphi",
            "sqrt_g",
            "B",
            "pressure",
        )
    }
    transfer_errors["force"] = _relative_error(refined_force.force, coarse_force.force)
    transfer_errors["force_max_absolute_N_per_m3"] = float(
        jnp.max(jnp.abs(refined_force.force - coarse_force.force))
    )
    if (
        spline_transfer_errors["derivative_0"] > 1.0e-12
        or spline_transfer_errors["derivative_1"] > 1.0e-10
        or spline_transfer_errors["derivative_2"] > 1.0e-6
        or max(
            value
            for name, value in transfer_errors.items()
            if name not in ("force", "force_max_absolute_N_per_m3")
        )
        > 1.0e-10
        or transfer_errors["force"] > 1.0e-6
    ):
        raise AssertionError(
            "exact refinement invariant failed: "
            f"splines={spline_transfer_errors}, fields={transfer_errors}"
        )

    ntheta = max(4 * int(np.max(np.abs(np.asarray(refined.m)))) + 5, 8)
    plan = make_variational_plan(refined, radial_order=4, ntheta=ntheta, nzeta=1)
    layout = make_native_correction_layout(refined)
    gauge = make_native_gauge_plan(refined, plan)
    coordinate_scale = native_coordinate_scales(refined, layout, plan)
    coordinates = jnp.zeros((layout.size,), dtype=jnp.float64)

    def force_residual(value):
        return native_physical_force_residual(
            value,
            refined,
            layout,
            gauge,
            coordinate_scale,
            force_scale,
            volume_scale,
        )

    def gauge_residual(value):
        return native_tangential_gauge_residual(
            refined,
            layout.unpack(coordinate_scale * value),
            gauge,
        )

    initial_force = force_residual(coordinates)
    force_rows = int(initial_force.size)
    memory = _dense_reference_memory(force_rows, layout.size, gauge.size)
    memory_limit = int(args.dense_memory_gib * 1024**3)
    if memory["explicit_array_bytes"] > memory_limit:
        raise MemoryError(
            "dense force plus one reduced Jacobian need "
            f"{memory['explicit_array_bytes']} bytes; cap is {memory_limit}"
        )
    constraint_started = time.perf_counter()
    constraint = np.asarray(jax.jacfwd(gauge_residual)(coordinates))
    q_range, nullspace, triangular, pivots = _constraint_qr(constraint)
    constraint_seconds = time.perf_counter() - constraint_started
    if nullspace.shape[1] != memory["expected_nullity"]:
        raise AssertionError("gauge nullity differs from the structural preflight")
    history = []
    jacobian_seconds = 0.0
    dense_solve_seconds = 0.0
    for iteration in range(args.max_steps):
        residual = np.asarray(force_residual(coordinates))
        defect = np.asarray(gauge_residual(coordinates))
        jacobian_started = time.perf_counter()
        jacobian = np.asarray(jax.jacfwd(force_residual)(coordinates))
        jacobian_seconds += time.perf_counter() - jacobian_started
        solve_started = time.perf_counter()
        reduced_jacobian = jacobian @ nullspace
        projected_gradient_norm = float(
            np.linalg.norm(reduced_jacobian.T @ residual)
        )
        projected_gradient_relative = projected_gradient_norm / max(
            np.linalg.norm(reduced_jacobian) * np.linalg.norm(residual),
            1.0e-300,
        )
        solve = _feasible_least_squares_step(
            jacobian,
            residual,
            constraint,
            defect,
            q_range,
            nullspace,
            triangular,
            pivots,
            reduced_jacobian=reduced_jacobian,
        )
        dense_solve_seconds += time.perf_counter() - solve_started
        trials = []
        accepted = None
        for fraction in (1.0, 0.5, 0.25, 0.125):
            candidate = coordinates + fraction * jnp.asarray(solve.step)
            candidate_norm = float(jnp.linalg.norm(force_residual(candidate)))
            candidate_gauge = float(jnp.linalg.norm(gauge_residual(candidate)))
            candidate_state = apply_high_order_correction(
                refined, layout.unpack(coordinate_scale * candidate)
            )
            minimum_j = float(minimum_signed_jacobian(candidate_state, plan))
            trial = {
                "fraction": fraction,
                "force_residual_norm": candidate_norm,
                "gauge_residual_norm": candidate_gauge,
                "minimum_signed_jacobian": minimum_j,
            }
            trials.append(trial)
            if (
                np.all(np.isfinite(tuple(trial.values())))
                and solve.rank == nullspace.shape[1]
                and solve.feasibility_norm < 1.0e-10
                and solve.projected_normal_relative < 1.0e-8
                and candidate_norm < float(np.linalg.norm(residual))
                and candidate_gauge < 1.0e-10
                and minimum_j > 0.0
            ):
                accepted = candidate
                break
        history.append(
            {
                "iteration": iteration + 1,
                "initial_force_residual_norm": float(np.linalg.norm(residual)),
                "constraint_defect_norm": float(np.linalg.norm(defect)),
                "reduced_jacobian_rank": solve.rank,
                "reduced_coordinates": int(nullspace.shape[1]),
                "linear_constraint_residual_norm": solve.feasibility_norm,
                "linearized_model_norm": solve.model_norm,
                "linearized_unreachable_residual_fraction": (
                    solve.model_norm / max(float(np.linalg.norm(residual)), 1.0e-300)
                ),
                "projected_normal_residual_norm": solve.projected_normal_norm,
                "projected_normal_residual_relative": solve.projected_normal_relative,
                "largest_reduced_singular_value": solve.largest_singular_value,
                "smallest_retained_reduced_singular_value": (
                    solve.smallest_retained_singular_value
                ),
                "projected_stationarity_norm": projected_gradient_norm,
                "projected_stationarity_relative": projected_gradient_relative,
                "accepted": accepted is not None,
                "trials": trials,
            }
        )
        if accepted is None:
            break
        coordinates = accepted

    accepted_state = apply_high_order_correction(
        refined, layout.unpack(coordinate_scale * coordinates)
    )
    certificate_started = time.perf_counter()
    certificate = certify_strong_force(accepted_state)
    certificate_seconds = time.perf_counter() - certificate_started
    final_projected, final_gradient, final_projected_relative = (
        _final_projected_gradient(force_residual, coordinates, nullspace)
    )
    arrays = _checkpoint_arrays(
        accepted_state,
        refined,
        coordinates,
        coordinate_scale,
        gauge,
    )
    _write_npz_atomic(args.output_state, arrays)
    output = {
        "schema": "vmex.polish-recovery/2",
        "experiment": "R3-exact-radial-h-refinement",
        "status": "measured-not-promoted",
        "source": {
            "input_native_state": str(args.input_state),
            "input_native_state_sha256": _sha256(args.input_state),
            "generator": str(Path(__file__).resolve().relative_to(Path.cwd())),
            "generator_sha256": _sha256(Path(__file__)),
            "output_native_state": str(args.output_state),
            "output_native_state_sha256": _sha256(args.output_state),
        },
        "transfer": {
            "method": "Boehm knot insertion through insert_high_order_state_knots",
            "coarse_spans": int(old_breakpoints.size - 1),
            "coarse_basis_size": int(coarse.radial_basis.size),
            "inserted_knots": inserted_knots.tolist(),
            "fine_spans": int(refined.radial_basis.breakpoints.size - 1),
            "fine_basis_size": int(refined.radial_basis.size),
            "selection": (
                "all span midpoints"
                if args.adaptive_count == 0
                else "largest volume-weighted force-squared span contributions"
            ),
            "selected_span_indices": selected_spans.tolist(),
            "span_force_squared_contributions": (
                None if span_scores is None else span_scores.tolist()
            ),
            "third_grid_relative_errors": transfer_errors,
            "spline_value_first_second_derivative_relative_errors": (
                spline_transfer_errors
            ),
            "seconds": transfer_seconds,
            "wout_refit": False,
        },
        "discretization": {
            "coordinates": layout.size,
            "gauge_constraints": gauge.size,
            "constraint_nullity": int(nullspace.shape[1]),
            "force_rows": force_rows,
            "ntheta": ntheta,
        },
        "checks": {
            "initial_force_residual_norm": float(jnp.linalg.norm(initial_force)),
            "history": history,
            "final_force_residual_norm": float(jnp.linalg.norm(force_residual(coordinates))),
            "final_projected_gradient_norm": final_projected,
            "final_full_gradient_norm": final_gradient,
            "final_projected_gradient_relative_to_full": final_projected_relative,
            "final_gauge_residual_norm": float(jnp.linalg.norm(gauge_residual(coordinates))),
            "independent_force_rms_N_per_m3": float(certificate.absolute_l2),
            "independent_epsilon_B": float(certificate.absolute_l2 / force_scale),
            "independent_radial_refinement_difference": float(
                certificate.radial_refinement_difference
            ),
            "minimum_signed_jacobian": float(certificate.minimum_signed_jacobian),
        },
        "work": {
            "explicit_jacobian_shape": [force_rows, layout.size],
            "explicit_jacobian_bytes": memory["full_jacobian_bytes"],
            "explicit_reduced_jacobian_bytes": memory[
                "one_reduced_jacobian_bytes"
            ],
            "explicit_array_budget_bytes": memory["explicit_array_bytes"],
            "dense_memory_cap_bytes": memory_limit,
            "constraint_factor_seconds": constraint_seconds,
            "jacobian_seconds": jacobian_seconds,
            "dense_solve_seconds": dense_solve_seconds,
            "certificate_seconds": certificate_seconds,
            "total_process_seconds": time.perf_counter() - started,
        },
        "limitations": [
            "This is a bounded dense reference on the axisymmetric user tokamak, not production linear algebra.",
            "One radial h-refinement does not qualify angular, degree, 3-D, current, derivative, or product gates.",
        ],
    }
    _write_json_atomic(args.output_json, output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
