"""Prospectively reproduce the bounded P3 Candidate B reference and checkpoint it."""

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
from scipy.linalg import lstsq, qr, solve_triangular

from vmex.core.input import VmecInput
from vmex.core.polish import apply_high_order_correction, make_native_correction_layout
from vmex.core.polish_variational import (
    make_native_gauge_plan,
    make_variational_plan,
    minimum_signed_jacobian,
    native_coordinate_scales,
    native_physical_force_residual,
    native_tangential_gauge_residual,
)
from vmex.core.strong_force import certify_strong_force, high_order_state_from_wout


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _constraint_qr(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a rank-checked orthogonal range/nullspace split of ``matrix``."""

    matrix = np.asarray(matrix, dtype=float)
    q, r, pivots = qr(matrix.T, mode="full", pivoting=True)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values.size == 0 or singular_values[-1] <= 1.0e-12 * singular_values[0]:
        raise ValueError("native gauge matrix is numerically rank deficient")
    rows = matrix.shape[0]
    return q[:, :rows], q[:, rows:], r[:rows], pivots


def _feasible_least_squares_step(
    jacobian: np.ndarray,
    residual: np.ndarray,
    constraint: np.ndarray,
    defect: np.ndarray,
    q_range: np.ndarray,
    nullspace: np.ndarray,
    triangular: np.ndarray,
    pivots: np.ndarray,
) -> tuple[np.ndarray, int, float, float]:
    """Solve one dense equality-constrained force least-squares reference step."""

    particular = q_range @ solve_triangular(
        triangular.T,
        -np.asarray(defect)[pivots],
        lower=True,
    )
    reduced = jacobian @ nullspace
    reduced_step, _, rank, _ = lstsq(
        reduced,
        -np.asarray(residual) - jacobian @ particular,
        cond=1.0e-12,
        lapack_driver="gelsd",
    )
    step = particular + nullspace @ reduced_step
    feasibility = float(np.linalg.norm(constraint @ step + defect))
    model = residual + jacobian @ step
    model_norm = float(np.linalg.norm(model))
    return step, int(rank), feasibility, model_norm


def _checkpoint_arrays(state, initial_state, coordinates, coordinate_scale, gauge):
    arrays = {
        "schema": np.asarray("vmex.polish-recovery-native-state/1"),
        "m": np.asarray(state.m),
        "n": np.asarray(state.n),
        "nfp": np.asarray(state.nfp),
        "jacobian_sign": np.asarray(state.jacobian_sign),
        "basis_knots": np.asarray(state.radial_basis.knots),
        "basis_breakpoints": np.asarray(state.radial_basis.breakpoints),
        "basis_degree": np.asarray(state.radial_basis.degree),
        "basis_periodic": np.asarray(state.radial_basis.periodic),
        "accepted_coordinates": np.asarray(coordinates),
        "coordinate_scale": np.asarray(coordinate_scale),
        "gauge_row_modes": np.asarray(gauge.row_modes),
        "gauge_row_basis": np.asarray(gauge.row_basis),
        "gauge_row_scale": np.asarray(gauge.row_scale),
    }
    for prefix, value in (("initial", initial_state), ("accepted", state)):
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
        ):
            arrays[f"{prefix}_{name}"] = np.asarray(getattr(value, name))
    return arrays


def _write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".npz", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


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
        "--input", type=Path, default=Path("examples/data/input.shaped_tokamak_pressure")
    )
    parser.add_argument(
        "--wout",
        type=Path,
        default=Path("artifacts/p0/run2/wout_shaped_tokamak_pressure.nc"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("benchmarks/polish_recovery_r1_p3.json"),
    )
    parser.add_argument(
        "--output-state",
        type=Path,
        default=Path("benchmarks/polish_recovery_r1_p3_state.npz"),
    )
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--dense-memory-gib", type=float, default=2.0)
    args = parser.parse_args()
    if args.max_steps < 1 or args.dense_memory_gib <= 0.0:
        parser.error("--max-steps and --dense-memory-gib must be positive")

    started = time.perf_counter()
    force_scale = 5915447.712414409
    volume_scale = 633.7993467060758
    inp = VmecInput.from_file(args.input)
    initial_state = high_order_state_from_wout(args.wout, inp=inp, degree=3)
    plan = make_variational_plan(initial_state, radial_order=4, ntheta=49, nzeta=1)
    layout = make_native_correction_layout(initial_state)
    gauge = make_native_gauge_plan(initial_state, plan)
    coordinate_scale = native_coordinate_scales(initial_state, layout, plan)
    coordinates = jnp.zeros((layout.size,), dtype=jnp.float64)

    def force_residual(value):
        return native_physical_force_residual(
            value,
            initial_state,
            layout,
            gauge,
            coordinate_scale,
            force_scale,
            volume_scale,
        )

    def gauge_residual(value):
        return native_tangential_gauge_residual(
            initial_state,
            layout.unpack(coordinate_scale * value),
            gauge,
        )

    initial_force = force_residual(coordinates)
    force_rows = int(initial_force.size)
    jacobian_bytes = force_rows * layout.size * np.dtype(np.float64).itemsize
    memory_limit = int(args.dense_memory_gib * 1024**3)
    if jacobian_bytes > memory_limit:
        raise MemoryError(
            f"dense force Jacobian needs {jacobian_bytes} bytes; cap is {memory_limit}"
        )
    constraint_started = time.perf_counter()
    constraint = np.asarray(jax.jacfwd(gauge_residual)(coordinates))
    q_range, nullspace, triangular, pivots = _constraint_qr(constraint)
    constraint_seconds = time.perf_counter() - constraint_started

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
        step, rank, feasibility, model_norm = _feasible_least_squares_step(
            jacobian,
            residual,
            constraint,
            defect,
            q_range,
            nullspace,
            triangular,
            pivots,
        )
        dense_solve_seconds += time.perf_counter() - solve_started
        trials = []
        accepted = None
        for fraction in (1.0, 0.5, 0.25, 0.125):
            candidate = coordinates + fraction * jnp.asarray(step)
            candidate_norm = float(jnp.linalg.norm(force_residual(candidate)))
            candidate_gauge = float(jnp.linalg.norm(gauge_residual(candidate)))
            corrected = apply_high_order_correction(
                initial_state,
                layout.unpack(coordinate_scale * candidate),
            )
            minimum_j = float(minimum_signed_jacobian(corrected, plan))
            trial = {
                "fraction": fraction,
                "force_residual_norm": candidate_norm,
                "gauge_residual_norm": candidate_gauge,
                "minimum_signed_jacobian": minimum_j,
            }
            trials.append(trial)
            if (
                np.all(np.isfinite(tuple(trial.values())))
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
                "reduced_jacobian_rank": rank,
                "reduced_coordinates": int(nullspace.shape[1]),
                "linear_constraint_residual_norm": feasibility,
                "linearized_model_norm": model_norm,
                "accepted": accepted is not None,
                "trials": trials,
            }
        )
        if accepted is None:
            break
        coordinates = accepted

    accepted_state = apply_high_order_correction(
        initial_state,
        layout.unpack(coordinate_scale * coordinates),
    )
    certificate_started = time.perf_counter()
    certificate = certify_strong_force(accepted_state)
    certificate_seconds = time.perf_counter() - certificate_started
    arrays = _checkpoint_arrays(
        accepted_state,
        initial_state,
        coordinates,
        coordinate_scale,
        gauge,
    )
    _write_npz_atomic(args.output_state, arrays)
    state_sha = _sha256(args.output_state)
    output = {
        "schema": "vmex.polish-recovery/2",
        "experiment": "R0-prospective-P3-Candidate-B-reproduction",
        "status": "measured-not-promoted",
        "source": {
            "input": str(args.input),
            "input_sha256": _sha256(args.input),
            "ordinary_wout": str(args.wout),
            "ordinary_wout_sha256": _sha256(args.wout),
            "generator": str(Path(__file__).resolve().relative_to(Path.cwd())),
            "generator_sha256": _sha256(Path(__file__)),
            "native_checkpoint": str(args.output_state),
            "native_checkpoint_sha256": state_sha,
        },
        "model": {
            "ncurr": int(inp.ncurr),
            "gamma": float(inp.gamma),
            "force_scale_N_per_m3": force_scale,
            "volume_scale_m3": volume_scale,
            "gauge_reference": "initial native state",
            "coordinate_scaling": "native_coordinate_scales saved in checkpoint",
        },
        "discretization": {
            "coordinates": layout.size,
            "gauge_constraints": gauge.size,
            "constraint_nullity": int(nullspace.shape[1]),
            "force_rows": force_rows,
            "basis_size": int(initial_state.radial_basis.size),
            "basis_degree": int(initial_state.radial_basis.degree),
            "radial_spans": int(initial_state.radial_basis.breakpoints.size - 1),
            "solve_radial_order": 4,
            "solve_ntheta": 49,
            "solve_nzeta": 1,
        },
        "checks": {
            "initial_force_residual_norm": float(jnp.linalg.norm(initial_force)),
            "history": history,
            "final_gauge_residual_norm": float(jnp.linalg.norm(gauge_residual(coordinates))),
            "final_force_residual_norm": float(jnp.linalg.norm(force_residual(coordinates))),
            "independent_force_rms_N_per_m3": float(certificate.absolute_l2),
            "independent_epsilon_B": float(certificate.absolute_l2 / force_scale),
            "independent_radial_refinement_difference": float(
                certificate.radial_refinement_difference
            ),
            "minimum_signed_jacobian": float(certificate.minimum_signed_jacobian),
        },
        "work": {
            "explicit_jacobian_shape": [force_rows, layout.size],
            "explicit_jacobian_bytes": jacobian_bytes,
            "dense_memory_cap_bytes": memory_limit,
            "constraint_factor_seconds": constraint_seconds,
            "jacobian_seconds": jacobian_seconds,
            "dense_solve_seconds": dense_solve_seconds,
            "certificate_seconds": certificate_seconds,
            "total_process_seconds": time.perf_counter() - started,
        },
        "limitations": [
            "This prospectively reconstructs the missing generator; it does not assert byte-for-byte identity with the historical untracked P3 script.",
            "The explicit force Jacobian and full QR nullspace are bounded reference algebra, not production linear algebra.",
            "Only the prescribed-iota axisymmetric user tokamak is exercised.",
        ],
    }
    _write_json_atomic(args.output_json, output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
