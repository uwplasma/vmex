"""Mode-block primal plus dense Schur reference preconditioner probe."""

from __future__ import annotations

import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from vmex.core.input import VmecInput
from vmex.core.polish import make_native_correction_layout
from vmex.core.polish_variational import (
    make_native_gauge_plan,
    make_variational_plan,
    native_coordinate_scales,
    native_force_gauss_newton_step,
    native_physical_force_residual,
    native_tangential_gauge_residual,
)
from vmex.core.radial_basis import BSplineBasis
from vmex.core.strong_force import high_order_state_from_wout


def main() -> None:
    started = time.perf_counter()
    force_scale = 5915447.712414409
    volume_scale = 633.7993467060758
    damping = 1.0e-2
    inp = VmecInput.from_file("examples/data/input.shaped_tokamak_pressure")
    basis = BSplineBasis.clamped(np.linspace(0.0, 1.0, 41), degree=3)
    state = high_order_state_from_wout(
        "artifacts/p0/run2/wout_shaped_tokamak_pressure.nc",
        inp=inp,
        radial_basis=basis,
        curvature_regularization=0.1,
    )
    plan = make_variational_plan(state, radial_order=4, ntheta=49, nzeta=1)
    layout = make_native_correction_layout(state)
    gauge = make_native_gauge_plan(state, plan)
    coordinate_scale = native_coordinate_scales(state, layout, plan)
    zero = jnp.zeros((layout.size,), dtype=jnp.float64)
    active = np.asarray(layout.active_indices, dtype=int)
    mode_ids = (active % (layout.mnmax * layout.nbasis)) // layout.nbasis
    groups = [active[mode_ids == mode] for mode in np.unique(mode_ids)]

    def local_force(local_coordinates, indices):
        full = zero.at[indices].set(local_coordinates)
        return native_physical_force_residual(
            full,
            state,
            layout,
            gauge,
            coordinate_scale,
            force_scale,
            volume_scale,
        )

    local_jacobian = jax.jit(jax.jacfwd(local_force, argnums=0))
    inverse_blocks = []
    block_sizes = []
    for indices in groups:
        indices_jax = jnp.asarray(indices, dtype=jnp.int32)
        jacobian = np.asarray(
            local_jacobian(jnp.zeros((indices.size,), dtype=jnp.float64), indices_jax)
        )
        block = jacobian.T @ jacobian + damping * np.eye(indices.size)
        inverse_blocks.append(np.linalg.inv(block))
        block_sizes.append(int(indices.size))

    def apply_primal(vector):
        result = jnp.zeros((layout.size,) + vector.shape[1:], dtype=vector.dtype)
        for indices, inverse in zip(groups, inverse_blocks, strict=True):
            indices_jax = jnp.asarray(indices, dtype=jnp.int32)
            value = jnp.asarray(inverse, dtype=vector.dtype) @ vector[indices_jax]
            result = result.at[indices_jax].set(value)
        return result

    def constraints(coordinates):
        return native_tangential_gauge_residual(
            state, layout.unpack(coordinate_scale * coordinates), gauge
        )

    constraint_matrix = np.asarray(jax.jacfwd(constraints)(zero))
    primal_inverse_constraint = np.asarray(
        apply_primal(jnp.asarray(constraint_matrix.T))
    )
    schur = constraint_matrix @ primal_inverse_constraint
    schur_condition = float(np.linalg.cond(schur))
    schur_inverse = np.linalg.inv(schur)

    def bordered_preconditioner(vector):
        primal_rhs = vector[: layout.size]
        constraint_rhs = vector[layout.size :]
        primal_intermediate = apply_primal(primal_rhs)
        multiplier = jnp.asarray(schur_inverse) @ (
            constraint_matrix @ primal_intermediate - constraint_rhs
        )
        primal = primal_intermediate - jnp.asarray(
            primal_inverse_constraint
        ) @ multiplier
        return jnp.concatenate((primal, multiplier))

    variables = jnp.zeros((layout.size + gauge.size,), dtype=jnp.float64)
    _, residual = native_force_gauss_newton_step(
        variables,
        state,
        layout,
        gauge,
        coordinate_scale,
        force_scale,
        volume_scale,
        damping=damping,
        tolerance=1.0e-4,
        restart=20,
        max_restarts=2,
        preconditioner=bordered_preconditioner,
    )
    output = {
        "coordinates": layout.size,
        "gauge_constraints": gauge.size,
        "mode_block_sizes": block_sizes,
        "block_hessian_bytes": int(sum(size * size for size in block_sizes) * 8),
        "materialized_constraint_matrix_bytes": int(constraint_matrix.nbytes),
        "dense_schur_condition": schur_condition,
        "restart": 20,
        "max_restarts": 2,
        "iteration_upper_bound": 40,
        "requested_true_relative_residual": 1.0e-4,
        "bordered_preconditioned_true_relative_residual": float(residual),
        "elapsed_seconds": time.perf_counter() - started,
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
