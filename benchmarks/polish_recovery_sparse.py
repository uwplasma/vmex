"""Recover one feasible native force step from compressed sparse derivatives.

This generator resumes a tracked native checkpoint.  It never constructs the
global dense force Jacobian or a dense gauge nullspace.  ``--insert-count 0``
uses the checkpoint's original frozen gauge; a positive count starts a declared
new exact-refinement stage from the accepted physical state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import tempfile
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from polish_recovery_p3 import _checkpoint_arrays, _write_npz_atomic
from polish_recovery_refine import _load_state
from solvax import CsrPattern, HostFactorOptions, factor_cache_info, sparse_solve
from solvax.compression import matrix_from_products, verify_products
from vmex.core.polish import apply_high_order_correction, make_native_correction_layout
from vmex.core.polish_variational import (
    evaluate_tensorized_strong_force,
    make_native_gauge_plan,
    make_variational_plan,
    minimum_signed_jacobian,
    native_coordinate_scales,
    native_force_jacobian_sparsity,
    native_physical_force_residual,
    native_tangential_gauge_matrix,
    native_tangential_gauge_residual,
)
from vmex.core.strong_force import (
    append_high_order_state_modes,
    certify_strong_force,
    evaluate_strong_force,
    insert_high_order_state_knots,
)


FORCE_SCALE = 5915447.712414409
VOLUME_SCALE = 633.7993467060758


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Darwin reports bytes; Linux and the BSD-compatible CI interface report KiB.
    return value if os.uname().sysname == "Darwin" else value * 1024


def _validate_scope(state) -> None:
    m = np.asarray(state.m, dtype=int)
    n = np.asarray(state.n, dtype=int)
    if state.radial_basis.periodic:
        raise ValueError("native force recovery requires an open radial basis")
    if m.ndim != 1 or n.shape != m.shape or m.size == 0:
        raise ValueError("native mode tables must be nonempty one-dimensional pairs")
    pairs = np.stack((m, n), axis=1)
    if np.unique(pairs, axis=0).shape[0] != pairs.shape[0]:
        raise ValueError("native mode table contains duplicate pairs")
    if np.any(n != 0) or not np.array_equal(m, np.arange(m.size)):
        raise NotImplementedError(
            "this measured sparse-recovery generator supports only the canonical axisymmetric m=0..mmax checkpoint"
        )
    if int(state.nfp) < 1 or int(state.jacobian_sign) not in (-1, 1):
        raise ValueError("checkpoint has invalid NFP or Jacobian orientation")
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
        if not np.all(np.isfinite(np.asarray(getattr(state, name)))):
            raise ValueError(f"checkpoint array {name} contains nonfinite values")


def _load_original_problem(path: Path):
    base = _load_state(path, prefix="initial")
    accepted = _load_state(path, prefix="accepted")
    _validate_scope(base)
    _validate_scope(accepted)
    with np.load(path, allow_pickle=False) as data:
        schema = str(np.asarray(data["schema"]).item())
        coordinates = np.asarray(data["accepted_coordinates"], dtype=float)
        saved_scale = np.asarray(data["coordinate_scale"], dtype=float)
        saved_modes = np.asarray(data["gauge_row_modes"], dtype=int)
        saved_basis = np.asarray(data["gauge_row_basis"], dtype=int)
        saved_row_scale = np.asarray(data["gauge_row_scale"], dtype=float)
        radial_order = (
            int(np.asarray(data["solve_radial_order"]).item()) if schema == "vmex.polish-recovery-native-state/2" else 4
        )
        ntheta = (
            int(np.asarray(data["solve_ntheta"]).item())
            if schema == "vmex.polish-recovery-native-state/2"
            else max(4 * int(np.max(np.abs(np.asarray(base.m)))) + 5, 8)
        )
        nzeta = int(np.asarray(data["solve_nzeta"]).item()) if schema == "vmex.polish-recovery-native-state/2" else 1
        if schema == "vmex.polish-recovery-native-state/2":
            if str(np.asarray(data["gauge_reference"]).item()) != "initial native state":
                raise ValueError("unsupported checkpoint gauge reference")
            np.testing.assert_allclose(float(np.asarray(data["force_scale_N_per_m3"]).item()), FORCE_SCALE)
            np.testing.assert_allclose(float(np.asarray(data["volume_scale_m3"]).item()), VOLUME_SCALE)
            if "input_sha256" in data:
                if str(np.asarray(data["input_sha256"]).item()) != (
                    "5b2740db6dbc91d6a1b43c283b95c959397a608566aeb54da7489203e1fe5102"
                ):
                    raise ValueError("checkpoint input hash is not the required deck")
                if (
                    float(np.asarray(data["gamma"]).item()) != 0.0
                    or int(np.asarray(data["ncurr"]).item()) != 0
                    or bool(np.asarray(data["lasym"]).item())
                ):
                    raise NotImplementedError("sparse recovery supports GAMMA=0, NCURR=0, LASYM=false")
    plan = make_variational_plan(base, radial_order=radial_order, ntheta=ntheta, nzeta=nzeta)
    layout = make_native_correction_layout(base)
    gauge = make_native_gauge_plan(base, plan)
    recomputed_scale = np.asarray(native_coordinate_scales(base, layout, plan))
    if coordinates.shape != (layout.size,) or saved_scale.shape != recomputed_scale.shape:
        raise ValueError("checkpoint coordinate vector or scale has the wrong shape")
    np.testing.assert_allclose(saved_scale, recomputed_scale, rtol=2.0e-13, atol=2.0e-13)
    # A same-chart replay uses the serialized metric exactly.  Even a
    # round-off-only contraction reorder can otherwise move large packed
    # coefficients by ~1e-11 in physical coefficient space.
    scale = saved_scale
    np.testing.assert_array_equal(saved_modes, np.asarray(gauge.row_modes))
    np.testing.assert_array_equal(saved_basis, np.asarray(gauge.row_basis))
    np.testing.assert_allclose(saved_row_scale, np.asarray(gauge.row_scale), rtol=2.0e-13, atol=2.0e-13)
    reconstructed = apply_high_order_correction(base, layout.unpack(jnp.asarray(scale * coordinates)))
    for name in ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin"):
        np.testing.assert_allclose(
            np.asarray(getattr(reconstructed, name)),
            np.asarray(getattr(accepted, name)),
            rtol=0.0,
            atol=2.0e-14,
        )
    return (
        base,
        accepted,
        plan,
        layout,
        gauge,
        jnp.asarray(scale),
        jnp.asarray(coordinates),
        radial_order,
    )


def _select_refinement(state, count: int, radial_order: int):
    if count < 1:
        return state, np.empty(0, dtype=int), np.empty(0), None
    indicator_order = max(int(radial_order) + 2, 6)
    plan = make_variational_plan(state, radial_order=indicator_order)
    samples = evaluate_tensorized_strong_force(state, plan)
    weighted = plan.quadrature_weights * state.jacobian_sign * samples.sqrt_g * jnp.sum(samples.force**2, axis=-1)
    span_count = state.radial_basis.breakpoints.size - 1
    scores = np.asarray(jnp.sum(weighted, axis=(1, 2))).reshape(span_count, indicator_order).sum(axis=1)
    if count >= span_count:
        raise ValueError("insert count must be smaller than the current span count")
    selected = np.sort(np.argsort(scores)[-count:])
    breaks = np.asarray(state.radial_basis.breakpoints)
    knots = 0.5 * (breaks[selected] + breaks[selected + 1])
    return insert_high_order_state_knots(state, knots), selected, knots, scores


def _linear_problem(base, plan, layout, gauge, scale, coordinates):
    def force(value):
        return native_physical_force_residual(value, base, layout, gauge, scale, FORCE_SCALE, VOLUME_SCALE)

    def constraint(value):
        return native_tangential_gauge_residual(base, layout.unpack(scale * value), gauge)

    return force, constraint


def _tensorized_certificate(state) -> dict[str, float | str]:
    """Fast overintegrated certificate with independent point spot checks."""

    maximum_m = int(np.max(np.abs(np.asarray(state.m))))
    maximum_n = int(np.max(np.abs(np.asarray(state.n))))
    ntheta = max(8, 4 * (maximum_m + 1))
    nzeta = max(4, 4 * (maximum_n + 1))

    def l2(order: int):
        plan = make_variational_plan(state, radial_order=order, ntheta=ntheta, nzeta=nzeta)
        samples = evaluate_tensorized_strong_force(state, plan)
        weights = plan.quadrature_weights * state.jacobian_sign * samples.sqrt_g
        magnitude_squared = jnp.sum(samples.force**2, axis=-1)
        value = jnp.sqrt(jnp.sum(weights * magnitude_squared) / jnp.sum(weights))
        signed_jacobian = state.jacobian_sign * samples.sqrt_g / jnp.maximum(plan.rho[:, None, None], 1.0e-14)
        return plan, samples, value, jnp.min(signed_jacobian)

    coarse_plan, _, coarse, _ = l2(6)
    fine_plan, fine_samples, fine, minimum_j = l2(8)
    radial_difference = jnp.abs(fine - coarse) / jnp.maximum(fine, 1.0e-12)
    radial_indices = np.linspace(0, fine_plan.shape[0] - 1, 12, dtype=int)
    theta_indices = (np.arange(12) * 7) % fine_plan.shape[1]
    zeta_indices = (np.arange(12) * 3) % fine_plan.shape[2]
    rho = fine_plan.rho[radial_indices]
    theta = fine_plan.theta[theta_indices]
    zeta = fine_plan.zeta[zeta_indices]
    oracle = evaluate_strong_force(state, rho, theta, zeta).force
    tensor = fine_samples.force[radial_indices, theta_indices, zeta_indices]
    point_error = jnp.linalg.norm(tensor - oracle) / jnp.maximum(jnp.linalg.norm(oracle), 1.0e-300)
    jax.block_until_ready((fine, radial_difference, minimum_j, point_error))
    return {
        "method": "overintegrated tensorized L2 with independent point-force spot checks",
        "force_rms_N_per_m3": float(fine),
        "epsilon_B": float(fine / FORCE_SCALE),
        "radial_refinement_difference": float(radial_difference),
        "minimum_signed_jacobian": float(minimum_j),
        "point_force_relative_error": float(point_error),
        "coarse_radial_order": int(coarse_plan.shape[0] // (state.radial_basis.breakpoints.size - 1)),
        "fine_radial_order": int(fine_plan.shape[0] // (state.radial_basis.breakpoints.size - 1)),
        "ntheta": ntheta,
        "nzeta": nzeta,
    }


def _compressed_jacobian(force, coordinates, structure):
    # Recompute each product instead of retaining jax.linearize's saved
    # intermediates.  The fixed-shape compiled JVP is reused across colors.
    apply = jax.jit(lambda direction: jax.jvp(force, (coordinates,), (direction,))[1])
    warm = apply(jnp.zeros_like(coordinates))
    jax.block_until_ready(warm)
    started = time.perf_counter()
    matrix = matrix_from_products(
        apply,
        structure.pattern,
        groups=structure.column_groups,
        dtype=np.float64,
    )
    seconds = time.perf_counter() - started
    error = verify_products(matrix, apply, samples=3, seed=448, dtype=np.float64)
    _, pullback = jax.vjp(force, coordinates)
    rng = np.random.default_rng(448)
    transpose_error = 0.0
    for _ in range(3):
        weight = rng.standard_normal(matrix.shape[0])
        expected = np.asarray(pullback(jnp.asarray(weight))[0])
        actual = np.asarray(matrix.T @ weight)
        transpose_error = max(
            transpose_error,
            float(np.linalg.norm(actual - expected)) / max(float(np.linalg.norm(expected)), np.finfo(float).tiny),
        )
    return matrix, seconds, error, transpose_error


def _sparse_feasible_step(matrix, residual, constraint, defect, damping: float):
    normal = (matrix.T @ matrix).tocsr()
    if damping:
        normal = normal + damping * sparse.eye(normal.shape[0], format="csr")
    kkt = sparse.bmat([[normal, constraint.T], [constraint, None]], format="csr")
    rhs = -np.concatenate((np.asarray(matrix.T @ residual), defect))
    pattern, values = CsrPattern.from_scipy(kkt, include_diagonal=True)
    # SOLVAX's SuperLU backend does not expose a native memory cap.  Admission
    # is handled before assembly and measured with process RSS; backend-level
    # memory controls are available only for optional MUMPS builds.
    options = HostFactorOptions(backend="superlu")
    started = time.perf_counter()
    solution = np.asarray(sparse_solve(pattern, jnp.asarray(values), jnp.asarray(rhs), options=options))
    jax.block_until_ready(solution)
    seconds = time.perf_counter() - started
    step = solution[: matrix.shape[1]]
    multipliers = solution[matrix.shape[1] :]
    model = residual + matrix @ step
    stationarity = matrix.T @ model + damping * step + constraint.T @ multipliers
    feasibility = constraint @ step + defect
    kkt_residual = np.concatenate((stationarity, feasibility))
    scale = max(float(np.linalg.norm(rhs)), np.finfo(float).tiny)
    projected_scale = max(float(np.linalg.norm(matrix.T @ residual)), np.finfo(float).tiny)
    return step, {
        "model_norm": float(np.linalg.norm(model)),
        "original_constraint_norm": float(np.linalg.norm(feasibility)),
        "stationarity_norm": float(np.linalg.norm(stationarity)),
        "stationarity_relative": float(np.linalg.norm(stationarity)) / projected_scale,
        "true_kkt_relative_residual": float(np.linalg.norm(kkt_residual)) / scale,
        "normal_nnz": int(normal.nnz),
        "kkt_nnz": int(kkt.nnz),
        "factor_solve_seconds": seconds,
        "factor_cache": factor_cache_info(),
    }


def _sparse_feasible_step_lsmr(
    matrix,
    residual,
    constraint,
    defect,
    damping: float,
    max_iterations: int,
):
    """Unsquared constrained least-squares reference with sparse projection."""

    coordinates = int(matrix.shape[1])
    projector_kkt = sparse.bmat(
        [
            [sparse.eye(coordinates, format="csr"), constraint.T],
            [constraint, None],
        ],
        format="csc",
    )
    started = time.perf_counter()
    factor = sparse_linalg.splu(projector_kkt)

    def projected(vector):
        rhs = np.concatenate((np.asarray(vector), np.zeros(constraint.shape[0], dtype=float)))
        return factor.solve(rhs)[:coordinates]

    particular = factor.solve(np.concatenate((np.zeros(coordinates), -np.asarray(defect))))[:coordinates]
    shifted = np.asarray(residual) + matrix @ particular
    operator = sparse_linalg.LinearOperator(
        matrix.shape,
        matvec=lambda value: matrix @ projected(value),
        rmatvec=lambda value: projected(matrix.T @ value),
        dtype=np.float64,
    )
    result = sparse_linalg.lsmr(
        operator,
        -shifted,
        damp=float(np.sqrt(damping)),
        atol=1.0e-11,
        btol=1.0e-11,
        conlim=1.0e12,
        maxiter=int(max_iterations),
    )
    step = particular + projected(result[0])
    seconds = time.perf_counter() - started
    model = np.asarray(residual) + matrix @ step
    gradient = matrix.T @ model + damping * step
    projected_gradient = projected(gradient)
    feasibility = constraint @ step + defect
    kkt_residual = np.concatenate((projected_gradient, feasibility))
    rhs_scale = max(float(np.linalg.norm(matrix.T @ residual)), np.finfo(float).tiny)
    return step, {
        "solver": "projected_unsquared_lsmr",
        "model_norm": float(np.linalg.norm(model)),
        "original_constraint_norm": float(np.linalg.norm(feasibility)),
        "stationarity_norm": float(np.linalg.norm(projected_gradient)),
        "stationarity_relative": float(np.linalg.norm(projected_gradient)) / rhs_scale,
        "true_kkt_relative_residual": float(np.linalg.norm(kkt_residual)) / rhs_scale,
        "factor_solve_seconds": seconds,
        "lsmr_stop_code": int(result[1]),
        "lsmr_iterations": int(result[2]),
        "lsmr_residual_norm": float(result[3]),
        "lsmr_normal_residual_norm": float(result[4]),
        "lsmr_operator_norm_estimate": float(result[5]),
        "lsmr_condition_estimate": float(result[6]),
        "projection_kkt_nnz": int(projector_kkt.nnz),
    }


def _sparse_projected_gradient(gradient, constraint):
    """Project a gradient through the original sparse gauge rows."""

    coordinates = int(constraint.shape[1])
    augmented = sparse.bmat(
        [
            [sparse.eye(coordinates, format="csr"), constraint.T],
            [constraint, None],
        ],
        format="csr",
    )
    rhs = np.concatenate((np.asarray(gradient), np.zeros(constraint.shape[0])))
    pattern, values = CsrPattern.from_scipy(augmented, include_diagonal=True)
    solution = np.asarray(
        sparse_solve(
            pattern,
            jnp.asarray(values),
            jnp.asarray(rhs),
            options=HostFactorOptions(backend="superlu"),
        )
    )
    projected = solution[:coordinates]
    multipliers = solution[coordinates:]
    stationarity = projected + constraint.T @ multipliers - gradient
    feasibility = constraint @ projected
    residual = np.concatenate((stationarity, feasibility))
    return projected, {
        "projection_true_residual_relative": float(np.linalg.norm(residual))
        / max(float(np.linalg.norm(rhs)), np.finfo(float).tiny),
        "projection_constraint_norm": float(np.linalg.norm(feasibility)),
        "projection_kkt_nnz": int(augmented.nnz),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-state",
        type=Path,
        default=Path("benchmarks/polish_recovery_r3_adaptive8_state.npz"),
    )
    parser.add_argument("--insert-count", type=int, default=0)
    parser.add_argument("--maximum-m", type=int)
    parser.add_argument("--radial-order", type=int, default=6)
    parser.add_argument(
        "--reanchor",
        action="store_true",
        help="start a declared new gauge/quadrature chart even without insertion",
    )
    parser.add_argument("--damping", type=float, default=0.0)
    parser.add_argument("--linear-solver", choices=("normal", "lsmr"), default="normal")
    parser.add_argument("--max-linear-iterations", type=int, default=80)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument(
        "--stationarity-refinement",
        action="store_true",
        help=(
            "after the force target is met, admit finite full-model steps that "
            "do not increase force beyond 1e-10 relative even when ordinary "
            "descent is below the numerical-floor gate"
        ),
    )
    parser.add_argument("--memory-gib", type=float, default=2.0)
    parser.add_argument("--certificate", choices=("point", "tensorized"), default="point")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("benchmarks/polish_recovery_r4_sparse.json"),
    )
    parser.add_argument(
        "--output-state",
        type=Path,
        default=Path("benchmarks/polish_recovery_r4_sparse_state.npz"),
    )
    args = parser.parse_args()
    if (
        args.insert_count < 0
        or args.max_steps < 0
        or args.radial_order < 2
        or args.damping < 0.0
        or args.max_linear_iterations < 1
    ):
        parser.error("counts and damping must be nonnegative")
    if args.memory_gib <= 0.0:
        parser.error("--memory-gib must be positive")

    started = time.perf_counter()
    (
        old_base,
        accepted,
        old_plan,
        old_layout,
        old_gauge,
        old_scale,
        old_coordinates,
        old_radial_order,
    ) = _load_original_problem(args.input_state)
    del old_layout, old_gauge, old_scale
    current_maximum_m = int(np.max(np.asarray(accepted.m)))
    angular_enrichment = args.maximum_m is not None and args.maximum_m > current_maximum_m
    if args.maximum_m is not None and args.maximum_m < current_maximum_m:
        parser.error("--maximum-m cannot remove checkpoint modes")
    if args.insert_count or args.reanchor or angular_enrichment:
        base, selected, inserted, scores = _select_refinement(accepted, args.insert_count, args.radial_order)
        if angular_enrichment:
            new_m = np.arange(current_maximum_m + 1, args.maximum_m + 1, dtype=int)
            base = append_high_order_state_modes(base, new_m, np.zeros_like(new_m))
        plan = make_variational_plan(base, radial_order=args.radial_order)
        layout = make_native_correction_layout(base)
        gauge = make_native_gauge_plan(base, plan)
        scale = native_coordinate_scales(base, layout, plan)
        coordinates = jnp.zeros((layout.size,), dtype=jnp.float64)
        stage = "new exact-refinement chart"
    else:
        base = old_base
        plan = old_plan
        layout = make_native_correction_layout(base)
        gauge = make_native_gauge_plan(base, plan)
        scale = native_coordinate_scales(base, layout, plan)
        coordinates = old_coordinates
        selected = np.empty(0, dtype=int)
        inserted = np.empty(0)
        scores = None
        stage = "saved original frozen gauge"

    force, constraint_function = _linear_problem(base, plan, layout, gauge, scale, coordinates)
    residual = np.asarray(force(coordinates))
    defect = np.asarray(constraint_function(coordinates))
    structure = native_force_jacobian_sparsity(layout, plan)
    estimated_sparse_bytes = int(
        structure.pattern.nnz * (np.dtype(np.float64).itemsize + np.dtype(np.int32).itemsize)
        + (structure.pattern.shape[0] + 1) * np.dtype(np.int64).itemsize
    )
    limit = int(args.memory_gib * 1024**3)
    if estimated_sparse_bytes > limit:
        raise MemoryError(
            f"sparse force values/indices need about {estimated_sparse_bytes} bytes; cap is {limit} before derivatives"
        )
    constraint_started = time.perf_counter()
    constraint = native_tangential_gauge_matrix(base, layout, gauge, scale)
    constraint_seconds = time.perf_counter() - constraint_started

    history = []
    assembly_seconds = 0.0
    solve_seconds = 0.0
    last_matrix = None
    last_linearization_coordinates = None
    for iteration in range(args.max_steps):
        residual = np.asarray(force(coordinates))
        defect = np.asarray(constraint_function(coordinates))
        matrix, elapsed, product_error, transpose_error = _compressed_jacobian(force, coordinates, structure)
        last_matrix = matrix
        last_linearization_coordinates = np.asarray(coordinates)
        assembly_seconds += elapsed
        if product_error > 2.0e-9 or transpose_error > 2.0e-9:
            raise AssertionError(f"compressed derivative mismatch: A={product_error}, AT={transpose_error}")
        if args.linear_solver == "normal":
            step, linear = _sparse_feasible_step(matrix, residual, constraint, defect, args.damping)
        else:
            step, linear = _sparse_feasible_step_lsmr(
                matrix,
                residual,
                constraint,
                defect,
                args.damping,
                args.max_linear_iterations,
            )
        solve_seconds += linear["factor_solve_seconds"]
        linear_passes = (
            linear["original_constraint_norm"] < 1.0e-10
            and linear["stationarity_relative"] < 1.0e-8
            and linear["true_kkt_relative_residual"] < 1.0e-8
        )
        trials = []
        accepted_coordinates = None
        acceptance_reason = None
        if linear_passes:
            for fraction in (1.0, 0.5, 0.25, 0.125):
                candidate = coordinates + fraction * jnp.asarray(step)
                candidate_force = float(jnp.linalg.norm(force(candidate)))
                candidate_gauge = float(jnp.linalg.norm(constraint_function(candidate)))
                candidate_state = apply_high_order_correction(base, layout.unpack(scale * candidate))
                minimum_j = float(minimum_signed_jacobian(candidate_state, plan))
                trial = {
                    "fraction": fraction,
                    "force_residual_norm": candidate_force,
                    "gauge_residual_norm": candidate_gauge,
                    "minimum_signed_jacobian": minimum_j,
                    "relative_force_reduction": (float(np.linalg.norm(residual)) - candidate_force)
                    / max(float(np.linalg.norm(residual)), np.finfo(float).tiny),
                }
                trials.append(trial)
                ordinary_descent = trial["relative_force_reduction"] > 1.0e-8
                stationarity_admission = (
                    args.stationarity_refinement
                    and float(np.linalg.norm(residual)) <= 1.0e-5
                    and trial["relative_force_reduction"] >= -1.0e-10
                )
                if (
                    np.all(np.isfinite(tuple(trial.values())))
                    and (ordinary_descent or stationarity_admission)
                    and candidate_gauge < 1.0e-10
                    and minimum_j > 0.0
                ):
                    accepted_coordinates = candidate
                    acceptance_reason = (
                        "ordinary_descent" if ordinary_descent else "force-qualified_stationarity_refinement"
                    )
                    break
        history.append(
            {
                "iteration": iteration + 1,
                "force_residual_norm": float(np.linalg.norm(residual)),
                "constraint_defect_norm": float(np.linalg.norm(defect)),
                "compressed_product_relative_error": product_error,
                "compressed_transpose_relative_error": transpose_error,
                "linear_certificate": linear,
                "linear_passes": linear_passes,
                "accepted": accepted_coordinates is not None,
                "acceptance_reason": acceptance_reason,
                "trials": trials,
            }
        )
        if accepted_coordinates is None:
            break
        coordinates = accepted_coordinates

    final_state = apply_high_order_correction(base, layout.unpack(scale * coordinates))
    arrays = _checkpoint_arrays(final_state, base, coordinates, scale, gauge)
    _write_npz_atomic(args.output_state, arrays)
    pending = {
        "schema": "vmex.polish-recovery/4",
        "experiment": "R5-compressed-sparse-feasible-force-step",
        "status": "candidate-saved-certification-pending",
        "complete": False,
        "product_qualified": False,
        "source": {
            "input_native_state": str(args.input_state),
            "input_native_state_sha256": _sha256(args.input_state),
            "generator": str(Path(__file__).resolve().relative_to(Path.cwd())),
            "generator_sha256": _sha256(Path(__file__)),
            "output_native_state": str(args.output_state),
            "output_native_state_sha256": _sha256(args.output_state),
        },
        "execution": {"phase": "certificate", "outcome": "running"},
        "certificate": {"complete": False, "force_pass": False},
        "stationarity": {"complete": False, "pass": False},
        "derivative": {"complete": False, "pass": False},
    }
    _write_json_atomic(args.output_json, pending)
    certificate_started = time.perf_counter()
    if args.certificate == "point":
        point = certify_strong_force(final_state)
        certificate = {
            "method": "independent shifted point oracle",
            "force_rms_N_per_m3": float(point.absolute_l2),
            "epsilon_B": float(point.absolute_l2 / FORCE_SCALE),
            "radial_refinement_difference": float(point.radial_refinement_difference),
            "minimum_signed_jacobian": float(point.minimum_signed_jacobian),
            "point_force_relative_error": 0.0,
        }
    else:
        certificate = _tensorized_certificate(final_state)
    certificate_seconds = time.perf_counter() - certificate_started
    final_residual, pullback = jax.vjp(force, coordinates)
    final_gradient = np.asarray(pullback(final_residual)[0])
    projected, projection = _sparse_projected_gradient(final_gradient, constraint)
    final_stationarity = float(np.linalg.norm(projected)) / max(
        float(np.linalg.norm(final_gradient)), np.finfo(float).tiny
    )
    final_stationarity_frobenius = None
    stationarity_assembly_seconds = 0.0
    if last_linearization_coordinates is not None and not np.array_equal(
        last_linearization_coordinates, np.asarray(coordinates)
    ):
        last_matrix, stationarity_assembly_seconds, product_error, transpose_error = _compressed_jacobian(
            force, coordinates, structure
        )
        if product_error > 2.0e-9 or transpose_error > 2.0e-9:
            raise AssertionError("final compressed derivative mismatch")
    if last_matrix is not None:
        operator_frobenius = float(np.linalg.norm(last_matrix.data))
        stationarity_reference = operator_frobenius * float(np.linalg.norm(np.asarray(final_residual)))
        final_stationarity_frobenius = float(np.linalg.norm(projected)) / max(
            stationarity_reference,
            np.finfo(float).tiny,
        )
    else:
        operator_frobenius = None
        stationarity_reference = None

    output = {
        "schema": "vmex.polish-recovery/4",
        "experiment": "R5-compressed-sparse-feasible-force-step",
        "status": "measured-not-promoted",
        "complete": True,
        "product_qualified": False,
        "source": {
            "input_native_state": str(args.input_state),
            "input_native_state_sha256": _sha256(args.input_state),
            "generator": str(Path(__file__).resolve().relative_to(Path.cwd())),
            "generator_sha256": _sha256(Path(__file__)),
            "output_native_state": str(args.output_state),
            "output_native_state_sha256": _sha256(args.output_state),
        },
        "model": {
            "stage": stage,
            "force_scale_N_per_m3": FORCE_SCALE,
            "volume_scale_m3": VOLUME_SCALE,
            "axisymmetric_only": True,
            "fixed_profiles": True,
            "damping": args.damping,
            "linear_solver": args.linear_solver,
            "max_linear_iterations": args.max_linear_iterations,
            "stationarity_refinement": args.stationarity_refinement,
            "solve_radial_order": (
                args.radial_order if args.insert_count or args.reanchor or angular_enrichment else old_radial_order
            ),
        },
        "refinement": {
            "selected_span_indices": selected.tolist(),
            "inserted_knots": inserted.tolist(),
            "span_force_squared_contributions": None if scores is None else scores.tolist(),
            "old_maximum_m": current_maximum_m,
            "new_maximum_m": int(np.max(np.asarray(base.m))),
        },
        "discretization": {
            "coordinates": layout.size,
            "gauge_rows": gauge.size,
            "force_rows": int(residual.size),
            "basis_size": int(base.radial_basis.size),
            "basis_degree": int(base.radial_basis.degree),
            "compressed_colors": len(structure.column_groups),
            "force_pattern_nnz": int(structure.pattern.nnz),
        },
        "checks": {
            "history": history,
            "final_force_residual_norm": float(jnp.linalg.norm(final_residual)),
            "final_gauge_residual_norm": float(jnp.linalg.norm(constraint_function(coordinates))),
            "final_projected_gradient_relative_to_full": final_stationarity,
            "final_projected_gradient_norm": float(np.linalg.norm(projected)),
            "final_full_gradient_norm": float(np.linalg.norm(final_gradient)),
            "final_operator_frobenius_norm": operator_frobenius,
            "stationarity_reference_norm": stationarity_reference,
            "final_projected_gradient_frobenius_relative": (final_stationarity_frobenius),
            "stationarity_tolerance": 1.0e-8,
            "stationarity_pass": (final_stationarity_frobenius is not None and final_stationarity_frobenius <= 1.0e-8),
            **projection,
            "force_certificate": certificate,
            "independent_force_rms_N_per_m3": (
                certificate["force_rms_N_per_m3"] if args.certificate == "point" else None
            ),
            "independent_epsilon_B": (certificate["epsilon_B"] if args.certificate == "point" else None),
            "independent_radial_refinement_difference": (
                certificate["radial_refinement_difference"] if args.certificate == "point" else None
            ),
            "minimum_signed_jacobian": certificate["minimum_signed_jacobian"],
        },
        "work": {
            "sparse_force_value_index_estimate_bytes": estimated_sparse_bytes,
            "memory_cap_bytes": limit,
            "peak_process_rss_bytes": _peak_rss_bytes(),
            "constraint_assembly_seconds": constraint_seconds,
            "compressed_force_assembly_seconds": assembly_seconds,
            "final_stationarity_assembly_seconds": stationarity_assembly_seconds,
            "sparse_factor_solve_seconds": solve_seconds,
            "certificate_seconds": certificate_seconds,
            "total_process_seconds": time.perf_counter() - started,
        },
        "limitations": [
            "This is an axisymmetric fixed-profile research generator, not the public polish driver.",
            "The sparse-direct normal equations require a conditioning-limited comparison before production promotion.",
            "Peak RSS is process high-water memory and may include earlier JAX compilation allocations.",
        ],
    }
    _write_json_atomic(args.output_json, output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
