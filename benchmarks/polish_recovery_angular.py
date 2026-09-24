"""Zero-pad angular modes and test their feasible force-recovery benefit."""

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
from polish_recovery_refine import _load_state
from vmex.core.polish import apply_high_order_correction, make_native_correction_layout
from vmex.core.polish_variational import (
    make_native_gauge_plan,
    make_variational_plan,
    minimum_signed_jacobian,
    native_coordinate_scales,
    native_physical_force_residual,
    native_tangential_gauge_residual,
)
from vmex.core.strong_force import append_high_order_state_modes, certify_strong_force


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        default=Path("benchmarks/polish_recovery_r3_refined_state.npz"),
    )
    parser.add_argument(
        "--output-state",
        type=Path,
        default=Path("benchmarks/polish_recovery_r3_angular_state.npz"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("benchmarks/polish_recovery_r3_angular.json"),
    )
    parser.add_argument("--maximum-m", type=int, default=15)
    parser.add_argument("--ntheta", type=int, default=65)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--dense-memory-gib", type=float, default=2.0)
    args = parser.parse_args()
    if args.max_steps < 1 or args.dense_memory_gib <= 0.0:
        parser.error("--max-steps and --dense-memory-gib must be positive")

    started = time.perf_counter()
    force_scale = 5915447.712414409
    volume_scale = 633.7993467060758
    base = _load_state(args.input_state)
    old_maximum_m = int(np.max(np.asarray(base.m)))
    if args.maximum_m <= old_maximum_m:
        parser.error("--maximum-m must exceed the checkpoint mode table")
    new_m = np.arange(old_maximum_m + 1, args.maximum_m + 1, dtype=int)
    enriched = append_high_order_state_modes(base, new_m, np.zeros_like(new_m))
    plan = make_variational_plan(
        enriched, radial_order=4, ntheta=args.ntheta, nzeta=1
    )
    layout = make_native_correction_layout(enriched)
    gauge = make_native_gauge_plan(enriched, plan)
    coordinate_scale = native_coordinate_scales(enriched, layout, plan)
    coordinates = jnp.zeros((layout.size,), dtype=jnp.float64)

    def force_residual(value):
        return native_physical_force_residual(
            value,
            enriched,
            layout,
            gauge,
            coordinate_scale,
            force_scale,
            volume_scale,
        )

    def gauge_residual(value):
        return native_tangential_gauge_residual(
            enriched,
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
        projected_gradient = float(
            np.linalg.norm(reduced_jacobian.T @ residual)
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
                enriched, layout.unpack(coordinate_scale * candidate)
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
                "projected_stationarity_norm": projected_gradient,
                "projected_normal_residual_norm": solve.projected_normal_norm,
                "projected_normal_residual_relative": solve.projected_normal_relative,
                "largest_reduced_singular_value": solve.largest_singular_value,
                "smallest_retained_reduced_singular_value": (
                    solve.smallest_retained_singular_value
                ),
                "accepted": accepted is not None,
                "trials": trials,
            }
        )
        if accepted is None:
            break
        coordinates = accepted

    accepted_state = apply_high_order_correction(
        enriched, layout.unpack(coordinate_scale * coordinates)
    )
    certificate_started = time.perf_counter()
    certificate = certify_strong_force(accepted_state)
    certificate_seconds = time.perf_counter() - certificate_started
    final_projected, final_gradient, final_projected_relative = (
        _final_projected_gradient(force_residual, coordinates, nullspace)
    )
    arrays = _checkpoint_arrays(
        accepted_state,
        enriched,
        coordinates,
        coordinate_scale,
        gauge,
    )
    _write_npz_atomic(args.output_state, arrays)
    output = {
        "schema": "vmex.polish-recovery/2",
        "experiment": "R3-angular-zero-padding-control",
        "status": "measured-not-promoted",
        "source": {
            "input_native_state": str(args.input_state),
            "input_native_state_sha256": _sha256(args.input_state),
            "generator": str(Path(__file__).resolve().relative_to(Path.cwd())),
            "generator_sha256": _sha256(Path(__file__)),
            "output_native_state": str(args.output_state),
            "output_native_state_sha256": _sha256(args.output_state),
        },
        "enrichment": {
            "old_maximum_m": old_maximum_m,
            "new_maximum_m": args.maximum_m,
            "new_modes": [[int(mode), 0] for mode in new_m],
            "transfer": "exact zero padding",
            "ntheta": args.ntheta,
        },
        "discretization": {
            "coordinates": layout.size,
            "gauge_constraints": gauge.size,
            "constraint_nullity": int(nullspace.shape[1]),
            "force_rows": force_rows,
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
            "This is a bounded dense axisymmetric control, not production linear algebra.",
            "Only four new axisymmetric angular modes and one or a few local steps are tested.",
        ],
    }
    _write_json_atomic(args.output_json, output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
