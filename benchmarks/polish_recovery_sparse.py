"""Recover one feasible native force step from compressed sparse derivatives.

This generator resumes a tracked native checkpoint.  It never constructs the
global dense force Jacobian or a dense gauge nullspace.  ``--insert-count 0``
uses the checkpoint's original frozen gauge; a positive count starts a declared
new exact-refinement stage from the accepted physical state.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
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

if __package__:
    from .polish_recovery_p3 import _checkpoint_arrays, _write_npz_atomic
    from .polish_recovery_refine import _load_state
else:
    from polish_recovery_p3 import _checkpoint_arrays, _write_npz_atomic
    from polish_recovery_refine import _load_state
from solvax import CsrPattern, HostFactorOptions, factor_cache_info, sparse_solve
from solvax.compression import matrix_from_products, verify_products
from vmex.core.polish import apply_high_order_correction, make_native_correction_layout
from vmex.core.polish_variational import (
    NativeGaugePlan,
    VariationalPlan,
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


STATIONARITY_TOLERANCE = 1.0e-8
# Maintainer decision 2026-09-25: the certified object in the accurate mode is
# the stored (initial state, accepted_coordinates) pair.  The rounded float64
# coefficient export is reported separately as representation-limited.
CERTIFIED_OBJECT = {
    "assembled-sum": "rounded float64 coefficient state",
    "coefficient-first-split-jets": "stored (initial state, accepted_coordinates) pair",
}
EVALUATION_MODES = ("assembled-sum", "coefficient-first-split-jets")


def _load_original_problem(path: Path, *, stable_derivatives: bool | None = None):
    """Load one saved chart exactly: base, plan, layout, gauge, stored metric.

    ``stable_derivatives=None`` replays the checkpoint's declared evaluation
    mode; True/False selects it explicitly (the chart and metric are unchanged).
    """

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
        saved_mode = str(np.asarray(data["evaluation_mode"]).item()) if "evaluation_mode" in data else EVALUATION_MODES[0]
        if saved_mode not in EVALUATION_MODES:
            raise ValueError(f"unknown checkpoint evaluation mode {saved_mode!r}")
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
    if stable_derivatives is None:
        stable_derivatives = saved_mode == EVALUATION_MODES[1]
    plan = make_variational_plan(
        base, radial_order=radial_order, ntheta=ntheta, nzeta=nzeta, stable_derivatives=bool(stable_derivatives)
    )
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


@jax.jit
def _batched_span_force_jvp(
    coordinates,
    directions,
    base_state,
    layout,
    gauge,
    coordinate_scale,
    force_scale,
    volume_scale,
    span_plan,
):
    """Apply colored force tangents on one radial slab with a shared kernel."""

    span_gauge = NativeGaugePlan(
        variational=span_plan,
        row_modes=gauge.row_modes,
        row_basis=gauge.row_basis,
        row_scale=gauge.row_scale,
        tangent_floor=gauge.tangent_floor,
    )

    def force(value):
        return native_physical_force_residual(
            value,
            base_state,
            layout,
            span_gauge,
            coordinate_scale,
            force_scale,
            volume_scale,
        )

    return jax.vmap(lambda direction: jax.jvp(force, (coordinates,), (direction,))[1])(directions)


def _radial_support(layout, plan):
    active = np.asarray(layout.active_indices, dtype=np.int64)
    radial_index = active % int(layout.nbasis)
    support = (
        (np.asarray(plan.radial_value) != 0.0)
        | (np.asarray(plan.radial_derivative) != 0.0)
        | (np.asarray(plan.radial_second_derivative) != 0.0)
    )
    return np.any(support, axis=0)[:, radial_index]


def _slice_variational_plan(plan: VariationalPlan, start: int, stop: int) -> VariationalPlan:
    """Return the same quadrature chart restricted to consecutive radial nodes."""

    return replace(
        plan,
        rho=plan.rho[start:stop],
        radial_value=plan.radial_value[:, start:stop, :],
        radial_derivative=plan.radial_derivative[:, start:stop, :],
        radial_second_derivative=plan.radial_second_derivative[:, start:stop, :],
        profile_basis=plan.profile_basis[start:stop, :],
        profile_derivative=plan.profile_derivative[start:stop, :],
        quadrature_weights=plan.quadrature_weights[start:stop, ...],
        **(
            {}
            if plan.spline_value is None
            else {
                "spline_value": plan.spline_value[start:stop],
                "spline_first": plan.spline_first[start:stop],
                "spline_second": plan.spline_second[start:stop],
                "axis_factors": plan.axis_factors[:, start:stop],
            }
        ),
    )


def _local_normal_system(force, coordinates, residual, base, plan, layout, gauge, scale):
    """Assemble the exact Gauss-Newton normal blocks one radial span at a time.

    Each temporary rectangular derivative contains only one span's residual
    rows.  It is discarded after contributing to H=A.T@A and g=A.T@r; the
    global force Jacobian and its 63-million-entry value array are never built.
    """

    try:
        from scipy import sparse
    except ImportError as error:  # pragma: no cover - SciPy is a VMEX dependency
        raise ImportError("local normal assembly requires SciPy") from error

    coordinates = np.asarray(coordinates, dtype=float)
    residual = np.asarray(residual, dtype=float)
    active = np.asarray(layout.active_indices, dtype=np.int64)
    basis_count = int(layout.nbasis)
    radial_index = active % basis_count
    field_mode = active // basis_count
    degree = int(base.radial_basis.degree)
    width = degree + 1
    _, color_of_column = np.unique(field_mode * width + radial_index % width, return_inverse=True)
    color_of_column = color_of_column.astype(np.int64, copy=False)
    color_count = int(np.max(color_of_column, initial=-1)) + 1
    support = _radial_support(layout, plan)
    radial_nodes = int(plan.rho.size)
    span_count = int(base.radial_basis.breakpoints.size - 1)
    if radial_nodes % span_count:
        raise ValueError("radial quadrature nodes are not an integer number per spline span")
    order = radial_nodes // span_count
    angular_components = int(plan.theta.size * plan.zeta.size * 3)
    row_block = order * angular_components

    directions = np.zeros((color_count, layout.size), dtype=np.float64)
    directions[color_of_column, np.arange(layout.size)] = 1.0
    directions_jax = jnp.asarray(directions)
    row_parts = []
    column_parts = []
    value_parts = []
    gradient = np.zeros(layout.size, dtype=np.float64)
    max_local_jacobian_nnz = 0
    verification_jvp_error = 0.0
    verification_vjp_error = 0.0
    started = time.perf_counter()

    for span in range(span_count):
        radial_start = span * order
        radial_stop = radial_start + order
        row_start = span * row_block
        row_stop = row_start + row_block
        radial_pattern = sparse.csr_matrix(support[radial_start:radial_stop].astype(np.int8))
        local_pattern = sparse.kron(
            radial_pattern,
            sparse.csr_matrix(np.ones((angular_components, 1), dtype=np.int8)),
            format="csr",
        )
        local_pattern.sort_indices()
        span_plan = _slice_variational_plan(plan, radial_start, radial_stop)
        products = np.asarray(
            _batched_span_force_jvp(
                jnp.asarray(coordinates),
                directions_jax,
                base,
                layout,
                gauge,
                scale,
                jnp.asarray(FORCE_SCALE),
                jnp.asarray(VOLUME_SCALE),
                span_plan,
            )
        )
        if products.shape != (color_count, row_block):
            raise ValueError(f"local compressed products have unexpected shape {products.shape}")
        row_indices = np.repeat(np.arange(row_block, dtype=np.int64), np.diff(local_pattern.indptr))
        values = products[color_of_column[local_pattern.indices], row_indices]
        local_jacobian = sparse.csr_matrix(
            (values, local_pattern.indices, local_pattern.indptr),
            shape=local_pattern.shape,
        )
        max_local_jacobian_nnz = max(max_local_jacobian_nnz, int(local_jacobian.nnz))
        # A span's Jacobian is nearly dense on its few active columns: gather
        # them, form the block with BLAS, and scatter it (sparse-sparse
        # products here took ~80% of the assembly time).
        active_columns = np.unique(local_pattern.indices)
        dense = local_jacobian[:, active_columns].toarray()
        block = dense.T @ dense
        row_parts.append(np.repeat(active_columns, active_columns.size))
        column_parts.append(np.tile(active_columns, active_columns.size))
        value_parts.append(block.reshape(-1))
        gradient[active_columns] += dense.T @ residual[row_start:row_stop]

    normal = sparse.coo_matrix(
        (np.concatenate(value_parts), (np.concatenate(row_parts), np.concatenate(column_parts))),
        shape=(layout.size, layout.size),
    ).tocsr()
    normal.sum_duplicates()
    normal.eliminate_zeros()
    asymmetry = normal - normal.T
    symmetry_error = float(np.linalg.norm(asymmetry.data)) / max(
        float(np.linalg.norm(normal.data)), np.finfo(float).tiny
    )
    if symmetry_error > 2.0e-12:
        raise ArithmeticError(f"local normal matrix is not symmetric: relative defect {symmetry_error}")

    # Compare assembled g and H v against independent full-domain JVP/VJP
    # actions.  These checks retain no rectangular global Jacobian.
    _, pullback = jax.vjp(force, jnp.asarray(coordinates))
    reference_gradient = np.asarray(pullback(jnp.asarray(residual))[0])
    verification_vjp_error = float(np.linalg.norm(gradient - reference_gradient)) / max(
        float(np.linalg.norm(reference_gradient)), np.finfo(float).tiny
    )
    probe = np.random.default_rng(448).standard_normal(layout.size)
    force_direction = np.asarray(jax.jvp(force, (jnp.asarray(coordinates),), (jnp.asarray(probe),))[1])
    reference_product = np.asarray(pullback(jnp.asarray(force_direction))[0])
    local_product = np.asarray(normal @ probe)
    verification_jvp_error = float(np.linalg.norm(local_product - reference_product)) / max(
        float(np.linalg.norm(reference_product)), np.finfo(float).tiny
    )
    if verification_vjp_error > 2.0e-9 or verification_jvp_error > 2.0e-9:
        raise AssertionError(
            f"local normal action mismatch: g={verification_vjp_error}, Hx={verification_jvp_error}"
        )
    return normal, gradient, {
        "assembly_seconds": time.perf_counter() - started,
        "normal_nnz": int(normal.nnz),
        "gradient_vjp_relative_error": verification_vjp_error,
        "normal_jvp_relative_error": verification_jvp_error,
        "symmetry_relative_error": symmetry_error,
        "max_local_jacobian_nnz": max_local_jacobian_nnz,
        "span_count": span_count,
        "color_count": color_count,
    }


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


def _sparse_normal_feasible_step(
    normal,
    gradient,
    force,
    coordinates,
    residual,
    constraint,
    defect,
    damping: float,
):
    """Solve a local-normal KKT system and certify it with full JVP/VJP actions."""

    normal = normal.tocsr()
    if damping:
        normal = normal + damping * sparse.eye(normal.shape[0], format="csr")
    kkt = sparse.bmat([[normal, constraint.T], [constraint, None]], format="csr")
    gradient = np.asarray(gradient, dtype=float)
    rhs = -np.concatenate((gradient, np.asarray(defect)))
    pattern, values = CsrPattern.from_scipy(kkt, include_diagonal=True)
    started = time.perf_counter()
    solution = np.asarray(
        sparse_solve(
            pattern,
            jnp.asarray(values),
            jnp.asarray(rhs),
            options=HostFactorOptions(backend="superlu"),
        )
    )
    jax.block_until_ready(solution)
    seconds = time.perf_counter() - started
    step = solution[: normal.shape[1]]
    multipliers = solution[normal.shape[1] :]

    force_direction = np.asarray(jax.jvp(force, (coordinates,), (jnp.asarray(step),))[1])
    linearized_residual = np.asarray(residual) + force_direction
    _, pullback = jax.vjp(force, coordinates)
    true_gradient = np.asarray(pullback(jnp.asarray(linearized_residual))[0])
    stationarity = true_gradient + damping * step + np.asarray(constraint.T @ multipliers)
    feasibility = np.asarray(constraint @ step + defect)
    kkt_residual = np.concatenate((stationarity, feasibility))
    rhs_scale = max(float(np.linalg.norm(rhs)), np.finfo(float).tiny)
    projected_scale = max(float(np.linalg.norm(gradient)), np.finfo(float).tiny)
    return step, {
        "model_norm": float(np.linalg.norm(linearized_residual)),
        "original_constraint_norm": float(np.linalg.norm(feasibility)),
        "stationarity_norm": float(np.linalg.norm(stationarity)),
        "stationarity_relative": float(np.linalg.norm(stationarity)) / projected_scale,
        "true_kkt_relative_residual": float(np.linalg.norm(kkt_residual)) / rhs_scale,
        "normal_nnz": int(normal.nnz),
        "kkt_nnz": int(kkt.nnz),
        "factor_solve_seconds": seconds,
        "factor_cache": factor_cache_info(),
        "certificate_method": "independent full-domain force JVP/VJP",
    }


def _matrix_free_preconditioned_lsmr_step(
    force,
    coordinates,
    residual,
    normal_diagonal,
    reference_gradient,
    constraint,
    defect,
    max_iterations: int,
):
    """Solve the equality-constrained unsquared step with diagonal right scaling."""

    coordinates = jnp.asarray(coordinates)
    residual = np.asarray(residual, dtype=float)
    diagonal = np.asarray(normal_diagonal, dtype=float)
    largest = float(np.max(diagonal, initial=0.0))
    if not np.isfinite(largest) or largest <= 0.0 or np.any(~np.isfinite(diagonal)):
        raise ValueError("local normal diagonal is not finite and positive")
    # Column-norm scaling is an explicit right preconditioner.  The floor is
    # recorded because nearly unobservable coordinates must not receive an
    # unbounded inverse scale.
    diagonal_floor = largest * 1.0e-16
    right_scale = np.sqrt(np.maximum(diagonal, diagonal_floor))
    inverse_scale = 1.0 / right_scale
    scaled_constraint = (constraint @ sparse.diags(inverse_scale, format="csr")).tocsr()
    projection_kkt = sparse.bmat(
        [
            [sparse.eye(constraint.shape[1], format="csr"), scaled_constraint.T],
            [scaled_constraint, None],
        ],
        format="csc",
    )
    started = time.perf_counter()
    factor = sparse_linalg.splu(projection_kkt)

    def project(vector):
        rhs = np.concatenate((np.asarray(vector), np.zeros(constraint.shape[0], dtype=float)))
        return factor.solve(rhs)[: constraint.shape[1]]

    particular_scaled = factor.solve(
        np.concatenate((np.zeros(constraint.shape[1], dtype=float), -np.asarray(defect)))
    )[: constraint.shape[1]]
    particular = inverse_scale * particular_scaled
    particular_force = np.asarray(
        jax.jvp(force, (coordinates,), (jnp.asarray(particular),))[1]
    )
    shifted = residual + particular_force
    _, pullback = jax.vjp(force, coordinates)

    def matvec(vector):
        feasible_scaled = project(vector)
        direction = inverse_scale * feasible_scaled
        return np.asarray(jax.jvp(force, (coordinates,), (jnp.asarray(direction),))[1])

    def rmatvec(vector):
        gradient = np.asarray(pullback(jnp.asarray(vector))[0])
        return project(inverse_scale * gradient)

    operator = sparse_linalg.LinearOperator(
        (residual.size, constraint.shape[1]),
        matvec=matvec,
        rmatvec=rmatvec,
        dtype=np.float64,
    )
    result = sparse_linalg.lsmr(
        operator,
        -shifted,
        atol=1.0e-11,
        btol=1.0e-11,
        conlim=1.0e12,
        maxiter=int(max_iterations),
    )
    step = particular + inverse_scale * project(result[0])
    model = residual + np.asarray(jax.jvp(force, (coordinates,), (jnp.asarray(step),))[1])
    physical_gradient = np.asarray(pullback(jnp.asarray(model))[0])
    projected_gradient, projection = _sparse_projected_gradient(physical_gradient, constraint)
    feasibility = np.asarray(constraint @ step + defect)
    kkt_residual = np.concatenate((projected_gradient, feasibility))
    gradient_scale = max(float(np.linalg.norm(reference_gradient)), np.finfo(float).tiny)
    seconds = time.perf_counter() - started
    return step, {
        "solver": "matrix_free_right_preconditioned_lsmr",
        "preconditioner": "sqrt_local_normal_diagonal",
        "preconditioner_diagonal_floor_relative": 1.0e-16,
        "preconditioner_scale_min": float(np.min(right_scale)),
        "preconditioner_scale_max": float(np.max(right_scale)),
        "model_norm": float(np.linalg.norm(model)),
        "original_constraint_norm": float(np.linalg.norm(feasibility)),
        "stationarity_norm": float(np.linalg.norm(projected_gradient)),
        "stationarity_relative": float(np.linalg.norm(projected_gradient))
        / gradient_scale,
        "true_kkt_relative_residual": float(np.linalg.norm(kkt_residual))
        / gradient_scale,
        "factor_solve_seconds": seconds,
        "lsmr_stop_code": int(result[1]),
        "lsmr_iterations": int(result[2]),
        "lsmr_residual_norm": float(result[3]),
        "lsmr_normal_residual_norm": float(result[4]),
        "lsmr_operator_norm_estimate": float(result[5]),
        "lsmr_condition_estimate": float(result[6]),
        "projection_kkt_nnz": int(projection_kkt.nnz),
        "projection_certificate": projection,
        "reference_gradient_norm": float(gradient_scale),
    }


def _exact_stationarity_step(
    force,
    coordinates,
    constraint,
    defect,
    gauss_newton_normal,
    max_iterations: int = 100,
):
    """Apply one exact constrained Newton correction to force least-squares stationarity.

    The exact Hessian-vector product includes the residual-weighted second
    derivative term.  The sparse Gauss-Newton KKT matrix is used only as a
    preconditioner; the solved equation and reported residual use the exact
    objective Hessian.
    """

    coordinates = jnp.asarray(coordinates)
    force_value = force(coordinates)

    def objective(value):
        residual = force(value)
        return 0.5 * jnp.vdot(residual, residual)

    gradient_function = jax.jit(jax.grad(objective))
    gradient = np.asarray(gradient_function(coordinates))
    hessian_action = jax.jit(
        lambda direction: jax.jvp(
            gradient_function,
            (coordinates,),
            (jnp.asarray(direction),),
        )[1]
    )
    coordinates_count = int(constraint.shape[1])
    variables_count = coordinates_count + int(constraint.shape[0])
    kkt_preconditioner = sparse.bmat(
        [[gauss_newton_normal, constraint.T], [constraint, None]],
        format="csc",
    )
    started = time.perf_counter()
    factor = sparse_linalg.splu(kkt_preconditioner)

    def apply(vector):
        coordinate_direction = np.asarray(vector[:coordinates_count])
        multiplier_direction = np.asarray(vector[coordinates_count:])
        primal = np.array(hessian_action(jnp.asarray(coordinate_direction)), copy=True)
        primal += np.asarray(constraint.T @ multiplier_direction)
        tangent = np.asarray(constraint @ coordinate_direction)
        return np.concatenate((primal, tangent))

    operator = sparse_linalg.LinearOperator(
        (variables_count, variables_count),
        matvec=apply,
        dtype=np.float64,
    )
    preconditioner = sparse_linalg.LinearOperator(
        (variables_count, variables_count),
        matvec=factor.solve,
        dtype=np.float64,
    )
    rhs = -np.concatenate((gradient, np.asarray(defect)))
    iterations = 0

    def count_iteration(_):
        nonlocal iterations
        iterations += 1

    solution, info = sparse_linalg.gmres(
        operator,
        rhs,
        M=preconditioner,
        rtol=1.0e-10,
        atol=0.0,
        restart=40,
        maxiter=int(max_iterations),
        callback=count_iteration,
        callback_type="pr_norm",
    )
    true_residual = apply(solution) - rhs
    step = solution[:coordinates_count]
    multipliers = solution[coordinates_count:]
    relative_residual = float(np.linalg.norm(true_residual)) / max(
        float(np.linalg.norm(rhs)), np.finfo(float).tiny
    )
    return step, {
        "solver": "exact_stationarity_newton_gmres",
        "preconditioner": "sparse_gauss_newton_kkt",
        "model_norm": float(np.linalg.norm(np.asarray(force_value))),
        "original_constraint_norm": float(np.linalg.norm(constraint @ step + defect)),
        "stationarity_norm": float(np.linalg.norm(true_residual[:coordinates_count])),
        "stationarity_relative": relative_residual,
        "true_kkt_relative_residual": relative_residual,
        "factor_solve_seconds": time.perf_counter() - started,
        "gmres_info": int(info),
        "gmres_iterations": iterations,
        "exact_hessian": True,
        "residual_weighted_hessian_term": True,
        "gradient_step_dot": float(np.vdot(gradient, step)),
        "step_norm": float(np.linalg.norm(step)),
        "kkt_nnz": int(kkt_preconditioner.nnz),
    }, gradient, multipliers, hessian_action, gradient_function


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


class _GaugeProjector:
    """One sparse LU of [[I, C.T], [C, 0]] per frozen chart.

    The factor is keyed on the constraint's exact CSR bytes, so a changed C
    always refactors; the same chart reuses one factor for every projection.
    """

    def __init__(self, constraint):
        constraint = sparse.csr_matrix(constraint)
        self.constraint = constraint
        self.key = _csr_key(constraint)
        self.coordinates = int(constraint.shape[1])
        self.augmented = sparse.bmat(
            [[sparse.eye(self.coordinates, format="csc"), constraint.T], [constraint, None]],
            format="csc",
        )
        self.factor = sparse_linalg.splu(self.augmented)
        self.solves = 0


def _csr_key(matrix) -> str:
    matrix = sparse.csr_matrix(matrix)
    digest = hashlib.sha256(str(matrix.shape).encode())
    for part in (matrix.data, matrix.indices, matrix.indptr):
        digest.update(np.ascontiguousarray(part).tobytes())
    return digest.hexdigest()


_PROJECTOR: dict[str, _GaugeProjector] = {}
PROJECTOR_STATS = {"factorizations": 0, "solves": 0}


def _gauge_projector(constraint) -> _GaugeProjector:
    key = _csr_key(constraint)
    projector = _PROJECTOR.get(key)
    if projector is None:
        _PROJECTOR.clear()  # one chart at a time; release the previous factor
        projector = _PROJECTOR[key] = _GaugeProjector(constraint)
        PROJECTOR_STATS["factorizations"] += 1
    return projector


def _sparse_projected_gradient(gradient, constraint):
    """Project a gradient through the original sparse gauge rows."""

    projector = _gauge_projector(constraint)
    constraint = projector.constraint
    coordinates = projector.coordinates
    rhs = np.concatenate((np.asarray(gradient, dtype=float), np.zeros(constraint.shape[0])))
    solution = projector.factor.solve(rhs)
    PROJECTOR_STATS["solves"] += 1
    projected = solution[:coordinates]
    multipliers = solution[coordinates:]
    stationarity = projected + constraint.T @ multipliers - rhs[:coordinates]
    feasibility = constraint @ projected
    residual = np.concatenate((stationarity, feasibility))
    return projected, {
        "projection_true_residual_relative": float(np.linalg.norm(residual))
        / max(float(np.linalg.norm(rhs)), np.finfo(float).tiny),
        "projection_constraint_norm": float(np.linalg.norm(feasibility)),
        "projection_kkt_nnz": int(projector.augmented.nnz),
        "projection_factorizations": PROJECTOR_STATS["factorizations"],
        "projection_solves": PROJECTOR_STATS["solves"],
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
    parser.add_argument("--linearization", choices=("sparse", "local-normal"), default="sparse")
    parser.add_argument("--max-linear-iterations", type=int, default=80)
    parser.add_argument("--stationarity-hessian", choices=("gauss-newton", "exact"), default="gauss-newton")
    parser.add_argument("--max-exact-newton-iterations", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument(
        "--stationarity-refinement",
        action="store_true",
        help=(
            "after the force target is met, enable bounded stationarity "
            "refinement; the exact-Hessian mode admits only projected-merit "
            "decreases that remain within the force target"
        ),
    )
    parser.add_argument(
        "--stable-derivatives",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "coefficient-first radial derivatives with base and correction jets "
            "synthesized separately (default: replay the checkpoint's mode)"
        ),
    )
    parser.add_argument(
        "--output-steps",
        type=Path,
        help="optional NPZ of raw (pre-line-search) steps and accepted fractions",
    )
    parser.add_argument("--memory-gib", type=float, default=2.0)
    parser.add_argument("--certificate", choices=("point", "tensorized", "none"), default="point")
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
        or args.max_exact_newton_iterations < 1
    ):
        parser.error("counts and damping must be nonnegative")
    if args.memory_gib <= 0.0:
        parser.error("--memory-gib must be positive")
    if args.linearization == "local-normal" and args.linear_solver == "lsmr" and args.damping:
        parser.error("matrix-free right-preconditioned LSMR currently requires zero damping")
    if args.stationarity_hessian == "exact" and (
        args.linearization != "local-normal" or args.linear_solver != "normal"
    ):
        parser.error("exact stationarity refinement requires --linearization local-normal and --linear-solver normal")

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
    ) = _load_original_problem(args.input_state, stable_derivatives=args.stable_derivatives)
    stable_derivatives = old_plan.spline_value is not None
    evaluation_mode = EVALUATION_MODES[int(stable_derivatives)]
    current_maximum_m = int(np.max(np.asarray(accepted.m)))
    angular_enrichment = args.maximum_m is not None and args.maximum_m > current_maximum_m
    if args.maximum_m is not None and args.maximum_m < current_maximum_m:
        parser.error("--maximum-m cannot remove checkpoint modes")
    if args.insert_count or args.reanchor or angular_enrichment:
        base, selected, inserted, scores = _select_refinement(accepted, args.insert_count, args.radial_order)
        if angular_enrichment:
            new_m = np.arange(current_maximum_m + 1, args.maximum_m + 1, dtype=int)
            base = append_high_order_state_modes(base, new_m, np.zeros_like(new_m))
        plan = make_variational_plan(base, radial_order=args.radial_order, stable_derivatives=stable_derivatives)
        layout = make_native_correction_layout(base)
        gauge = make_native_gauge_plan(base, plan)
        scale = native_coordinate_scales(base, layout, plan)
        coordinates = jnp.zeros((layout.size,), dtype=jnp.float64)
        stage = "new exact-refinement chart"
    else:
        # Same-chart continuation: reuse the validated objects and the exact
        # serialized metric.  Recomputing the scale here can differ by
        # round-off and silently move the physical coefficients.
        base = old_base
        plan = old_plan
        layout = old_layout
        gauge = old_gauge
        scale = old_scale
        coordinates = old_coordinates
        selected = np.empty(0, dtype=int)
        inserted = np.empty(0)
        scores = None
        stage = "saved original frozen gauge"

    force, constraint_function = _linear_problem(base, plan, layout, gauge, scale, coordinates)
    residual = np.asarray(force(coordinates))
    defect = np.asarray(constraint_function(coordinates))
    structure = None
    local_support = None
    local_color_count = None
    global_pattern_nnz = None
    if args.linearization == "sparse":
        structure = native_force_jacobian_sparsity(layout, plan)
        global_pattern_nnz = int(structure.pattern.nnz)
        estimated_sparse_bytes = int(
            structure.pattern.nnz * (np.dtype(np.float64).itemsize + np.dtype(np.int32).itemsize)
            + (structure.pattern.shape[0] + 1) * np.dtype(np.int64).itemsize
        )
        linearization_estimate_name = "global_sparse_jacobian_values_and_indices"
    else:
        local_support = _radial_support(layout, plan)
        degree = int(base.radial_basis.degree)
        width = degree + 1
        active = np.asarray(layout.active_indices)
        raw_colors = (active // int(layout.nbasis)) * width + (active % int(layout.nbasis)) % width
        local_color_count = int(np.unique(raw_colors).size)
        span_count = int(base.radial_basis.breakpoints.size - 1)
        order = int(plan.rho.size) // span_count
        angular_components = int(plan.theta.size * plan.zeta.size * 3)
        local_jacobian_nnz = 0
        local_normal_contribution_nnz = 0
        for span in range(span_count):
            start = span * order
            stop = start + order
            local_jacobian_nnz = max(
                local_jacobian_nnz,
                int(np.sum(local_support[start:stop])) * angular_components,
            )
            local_columns = int(np.count_nonzero(np.any(local_support[start:stop], axis=0)))
            local_normal_contribution_nnz += local_columns * local_columns
        global_pattern_nnz = int(np.sum(local_support)) * angular_components
        estimated_sparse_bytes = int(
            local_jacobian_nnz * (np.dtype(np.float64).itemsize + np.dtype(np.int32).itemsize)
            + local_normal_contribution_nnz * (np.dtype(np.float64).itemsize + 2 * np.dtype(np.int64).itemsize)
        )
        linearization_estimate_name = "largest_local_jacobian_plus_all_local_normal_contributions"
    limit = int(args.memory_gib * 1024**3)
    if estimated_sparse_bytes > limit:
        raise MemoryError(
            f"{linearization_estimate_name} needs about {estimated_sparse_bytes} bytes; "
            f"cap is {limit} before derivatives"
        )
    constraint_started = time.perf_counter()
    constraint = native_tangential_gauge_matrix(base, layout, gauge, scale)
    constraint_seconds = time.perf_counter() - constraint_started

    history = []
    assembly_seconds = 0.0
    solve_seconds = 0.0
    last_matrix = None
    last_normal = None
    last_local_metrics = None
    last_linearization_coordinates = None
    raw_steps = []
    for iteration in range(args.max_steps):
        residual = np.asarray(force(coordinates))
        defect = np.asarray(constraint_function(coordinates))
        if args.linearization == "sparse":
            matrix, elapsed, product_error, transpose_error = _compressed_jacobian(force, coordinates, structure)
            last_matrix = matrix
            last_normal = None
            last_local_metrics = None
        else:
            normal, gradient, local_metrics = _local_normal_system(
                force, coordinates, residual, base, plan, layout, gauge, scale
            )
            elapsed = local_metrics["assembly_seconds"]
            product_error = local_metrics["normal_jvp_relative_error"]
            transpose_error = local_metrics["gradient_vjp_relative_error"]
            last_matrix = None
            last_normal = normal
            last_local_metrics = local_metrics
        last_linearization_coordinates = np.asarray(coordinates)
        assembly_seconds += elapsed
        if product_error > 2.0e-9 or transpose_error > 2.0e-9:
            raise AssertionError(f"linearization mismatch: forward={product_error}, transpose={transpose_error}")
        exact_stationarity = (
            args.stationarity_hessian == "exact"
            and args.stationarity_refinement
            and float(np.linalg.norm(residual)) <= 1.0e-5
        )
        exact_gradient = None
        exact_hessian_action = None
        exact_gradient_function = None
        current_exact_stationarity = None
        exact_projection_metrics = None
        if exact_stationarity:
            (
                step,
                linear,
                exact_gradient,
                _,
                exact_hessian_action,
                exact_gradient_function,
            ) = _exact_stationarity_step(
                force,
                coordinates,
                constraint,
                defect,
                normal,
                max_iterations=args.max_exact_newton_iterations,
            )
            projected_gradient, exact_projection_metrics = _sparse_projected_gradient(exact_gradient, constraint)
            if exact_projection_metrics["projection_true_residual_relative"] > 1.0e-10:
                raise ArithmeticError("exact-stationarity gradient projection failed its true-residual check")
            current_operator_frobenius = float(np.sqrt(max(float(np.sum(normal.diagonal())), 0.0)))
            current_exact_stationarity = float(np.linalg.norm(projected_gradient)) / max(
                current_operator_frobenius * float(np.linalg.norm(residual)), np.finfo(float).tiny
            )
            if current_exact_stationarity <= STATIONARITY_TOLERANCE:
                # Converged: further steps are below the evaluation floor and
                # would only select among rounding-level candidates.
                history.append(
                    {
                        "iteration": iteration + 1,
                        "force_residual_norm": float(np.linalg.norm(residual)),
                        "current_exact_stationarity_frobenius": current_exact_stationarity,
                        "accepted": False,
                        "acceptance_reason": "stationarity_target_met",
                    }
                )
                break
        elif args.linear_solver == "normal":
            if args.linearization == "sparse":
                step, linear = _sparse_feasible_step(matrix, residual, constraint, defect, args.damping)
            else:
                step, linear = _sparse_normal_feasible_step(
                    normal,
                    gradient,
                    force,
                    jnp.asarray(coordinates),
                    residual,
                    constraint,
                    defect,
                    args.damping,
                )
        elif args.linearization == "local-normal":
            step, linear = _matrix_free_preconditioned_lsmr_step(
                force,
                jnp.asarray(coordinates),
                residual,
                np.asarray(normal.diagonal()),
                gradient,
                constraint,
                defect,
                args.max_linear_iterations,
            )
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
        exact_gradient_fallback = None
        if linear_passes:
            for fraction in (1.0, 0.5, 0.25, 0.125):
                candidate = coordinates + fraction * jnp.asarray(step)
                candidate_residual = np.asarray(force(candidate))
                candidate_force = float(np.linalg.norm(candidate_residual))
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
                stationarity_merit_improved = False
                if (
                    exact_stationarity
                    and candidate_force <= 1.0e-5
                    and candidate_gauge < 1.0e-10
                    and minimum_j > 0.0
                ):
                    candidate_gradient = np.asarray(exact_gradient_function(candidate))
                    candidate_projected, candidate_projection = _sparse_projected_gradient(
                        candidate_gradient, constraint
                    )
                    candidate_stationarity = float(np.linalg.norm(candidate_projected)) / max(
                        current_operator_frobenius * candidate_force, np.finfo(float).tiny
                    )
                    trial["exact_stationarity_frobenius"] = candidate_stationarity
                    trial["stationarity_projection_true_residual_relative"] = candidate_projection[
                        "projection_true_residual_relative"
                    ]
                    stationarity_merit_improved = (
                        candidate_projection["projection_true_residual_relative"] <= 1.0e-10
                        and candidate_stationarity < current_exact_stationarity * (1.0 - 1.0e-4)
                    )
                trials.append(trial)
                ordinary_descent = trial["relative_force_reduction"] > 1.0e-8
                stationarity_admission = (
                    args.stationarity_refinement
                    and float(np.linalg.norm(residual)) <= 1.0e-5
                    and trial["relative_force_reduction"] >= -1.0e-10
                )
                if (
                    np.all(np.isfinite(tuple(trial.values())))
                    and (
                        stationarity_merit_improved
                        if exact_stationarity
                        else (ordinary_descent or stationarity_admission)
                    )
                    and candidate_gauge < 1.0e-10
                    and minimum_j > 0.0
                ):
                    accepted_coordinates = candidate
                    acceptance_reason = (
                        "stationarity_merit_reduction_within_force_gate"
                        if exact_stationarity
                        else ("ordinary_descent" if ordinary_descent else "force-qualified_stationarity_refinement")
                    )
                    break
        accepted_fraction = trials[-1]["fraction"] if accepted_coordinates is not None and trials else None
        raw_steps.append((np.asarray(step, dtype=float), accepted_fraction))
        if accepted_coordinates is None and exact_stationarity:
            projected_gradient, _ = _sparse_projected_gradient(exact_gradient, constraint)
            projected_norm = float(np.linalg.norm(projected_gradient))
            curvature = float(
                np.vdot(
                    projected_gradient,
                    np.asarray(exact_hessian_action(jnp.asarray(projected_gradient))),
                )
            )
            if curvature > 0.0 and np.isfinite(curvature):
                gradient_step_scale = float(np.vdot(projected_gradient, projected_gradient)) / curvature
                gradient_step_scale = min(
                    gradient_step_scale,
                    float(np.linalg.norm(step)) / max(projected_norm, np.finfo(float).tiny),
                )
            else:
                gradient_step_scale = 0.0
            exact_gradient_fallback = {
                "projected_gradient_norm": projected_norm,
                "projected_gradient_curvature": curvature,
                "quadratic_step_scale": gradient_step_scale,
            }
            for fraction in (2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125):
                candidate = coordinates - (fraction * gradient_step_scale) * jnp.asarray(projected_gradient)
                candidate_residual = np.asarray(force(candidate))
                candidate_force = float(np.linalg.norm(candidate_residual))
                candidate_gauge = float(jnp.linalg.norm(constraint_function(candidate)))
                candidate_state = apply_high_order_correction(base, layout.unpack(scale * candidate))
                minimum_j = float(minimum_signed_jacobian(candidate_state, plan))
                trial = {
                    "direction": "projected_negative_gradient",
                    "fraction": fraction,
                    "force_residual_norm": candidate_force,
                    "gauge_residual_norm": candidate_gauge,
                    "minimum_signed_jacobian": minimum_j,
                    "relative_force_reduction": (float(np.linalg.norm(residual)) - candidate_force)
                    / max(float(np.linalg.norm(residual)), np.finfo(float).tiny),
                }
                stationarity_merit_improved = False
                if candidate_force <= 1.0e-5 and candidate_gauge < 1.0e-10 and minimum_j > 0.0:
                    candidate_gradient = np.asarray(exact_gradient_function(candidate))
                    candidate_projected, candidate_projection = _sparse_projected_gradient(
                        candidate_gradient, constraint
                    )
                    candidate_stationarity = float(np.linalg.norm(candidate_projected)) / max(
                        current_operator_frobenius * candidate_force, np.finfo(float).tiny
                    )
                    trial["exact_stationarity_frobenius"] = candidate_stationarity
                    trial["stationarity_projection_true_residual_relative"] = candidate_projection[
                        "projection_true_residual_relative"
                    ]
                    stationarity_merit_improved = (
                        candidate_projection["projection_true_residual_relative"] <= 1.0e-10
                        and candidate_stationarity < current_exact_stationarity * (1.0 - 1.0e-4)
                    )
                trials.append(trial)
                if (
                    np.all(np.isfinite(tuple(value for key, value in trial.items() if key != "direction")))
                    and stationarity_merit_improved
                    and candidate_force <= 1.0e-5
                    and candidate_gauge < 1.0e-10
                    and minimum_j > 0.0
                ):
                    accepted_coordinates = candidate
                    acceptance_reason = "stationarity_merit_gradient_fallback_within_force_gate"
                    break
        history.append(
            {
                "iteration": iteration + 1,
                "force_residual_norm": float(np.linalg.norm(residual)),
                "constraint_defect_norm": float(np.linalg.norm(defect)),
                "compressed_product_relative_error": product_error,
                "compressed_transpose_relative_error": transpose_error,
                "local_normal_assembly": last_local_metrics,
                "linear_certificate": linear,
                "linear_passes": linear_passes,
                "current_exact_stationarity_frobenius": current_exact_stationarity,
                "exact_stationarity_projection": exact_projection_metrics,
                "exact_hessian_gradient_fallback": exact_gradient_fallback,
                "accepted": accepted_coordinates is not None,
                "acceptance_reason": acceptance_reason,
                "raw_step_norm": float(np.linalg.norm(raw_steps[-1][0])),
                "raw_step_accepted_fraction": raw_steps[-1][1],
                "trials": trials,
            }
        )
        if accepted_coordinates is None:
            break
        coordinates = accepted_coordinates

    final_state = apply_high_order_correction(base, layout.unpack(scale * coordinates))
    arrays = _checkpoint_arrays(final_state, base, coordinates, scale, gauge)
    arrays["evaluation_mode"] = np.asarray(evaluation_mode)
    _write_npz_atomic(args.output_state, arrays)
    if args.output_steps is not None:
        _write_npz_atomic(
            args.output_steps,
            {
                "raw_steps": np.stack([step for step, _ in raw_steps]) if raw_steps else np.zeros((0, layout.size)),
                "accepted_fraction": np.asarray([np.nan if f is None else f for _, f in raw_steps]),
                "input_state_sha256": np.asarray(_sha256(args.input_state)),
            },
        )
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
    elif args.certificate == "tensorized":
        certificate = _tensorized_certificate(final_state)
    else:
        # Intermediate continuation stage: no independent force claim is made.
        certificate = {
            "method": "skipped (intermediate stage; in-loop quadrature norm only)",
            "minimum_signed_jacobian": float(minimum_signed_jacobian(final_state, plan)),
        }
    certificate_seconds = time.perf_counter() - certificate_started
    final_residual, pullback = jax.vjp(force, coordinates)
    final_gradient = np.asarray(pullback(final_residual)[0])
    projected, projection = _sparse_projected_gradient(final_gradient, constraint)
    final_stationarity = float(np.linalg.norm(projected)) / max(
        float(np.linalg.norm(final_gradient)), np.finfo(float).tiny
    )
    final_stationarity_frobenius = None
    stationarity_assembly_seconds = 0.0
    intermediate = args.certificate == "none"
    if intermediate:
        # No stationarity claim is made for an intermediate stage; skip the
        # final re-linearization rather than normalize with a stale operator.
        last_matrix = last_normal = None
    if (
        not intermediate
        and last_linearization_coordinates is not None
        and not np.array_equal(last_linearization_coordinates, np.asarray(coordinates))
    ):
        if args.linearization == "sparse":
            last_matrix, stationarity_assembly_seconds, product_error, transpose_error = _compressed_jacobian(
                force, coordinates, structure
            )
        else:
            last_normal, _, last_local_metrics = _local_normal_system(
                force, coordinates, np.asarray(final_residual), base, plan, layout, gauge, scale
            )
            stationarity_assembly_seconds = last_local_metrics["assembly_seconds"]
            product_error = last_local_metrics["normal_jvp_relative_error"]
            transpose_error = last_local_metrics["gradient_vjp_relative_error"]
        if product_error > 2.0e-9 or transpose_error > 2.0e-9:
            raise AssertionError("final linearization mismatch")
    if last_matrix is not None:
        operator_frobenius = float(np.linalg.norm(last_matrix.data))
        stationarity_reference = operator_frobenius * float(np.linalg.norm(np.asarray(final_residual)))
        final_stationarity_frobenius = float(np.linalg.norm(projected)) / max(
            stationarity_reference,
            np.finfo(float).tiny,
        )
    elif last_normal is not None:
        operator_frobenius = float(np.sqrt(max(float(np.sum(last_normal.diagonal())), 0.0)))
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
            "linearization": args.linearization,
            "stationarity_hessian": args.stationarity_hessian,
            "evaluation_mode": evaluation_mode,
            "max_exact_newton_iterations": args.max_exact_newton_iterations,
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
            "compressed_colors": (
                len(structure.column_groups) if structure is not None else local_color_count
            ),
            "force_pattern_nnz": global_pattern_nnz,
            "global_force_jacobian_materialized": structure is not None,
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
            "stationarity_tolerance": STATIONARITY_TOLERANCE,
            "stationarity_certified_object": CERTIFIED_OBJECT[evaluation_mode],
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
            "linearization_memory_estimate_name": linearization_estimate_name,
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
