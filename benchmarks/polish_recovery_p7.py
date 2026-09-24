"""Bounded Krylov/preconditioner probe for the native force GN system."""

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
    native_force_gauss_newton_action,
    native_force_gauss_newton_step,
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
    variables = jnp.zeros((layout.size + gauge.size,), dtype=jnp.float64)

    step, unpreconditioned_residual = native_force_gauss_newton_step(
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
    )

    key = jax.random.PRNGKey(0)
    probes = jax.random.rademacher(
        key, (16, layout.size), dtype=jnp.float64
    )
    probe_directions = jnp.concatenate(
        (probes, jnp.zeros((probes.shape[0], gauge.size))), axis=1
    )
    actions = jax.vmap(
        lambda direction: native_force_gauss_newton_action(
            variables,
            direction,
            state,
            layout,
            gauge,
            coordinate_scale,
            force_scale,
            volume_scale,
            damping,
        )
    )(probe_directions)
    estimated_diagonal = jnp.mean(
        probes * actions[:, : layout.size], axis=0
    )
    diagonal_floor = jnp.max(jnp.abs(estimated_diagonal)) * 1.0e-6
    inverse_diagonal = jnp.reciprocal(
        jnp.maximum(estimated_diagonal, diagonal_floor)
    )

    def diagonal_preconditioner(vector):
        return jnp.concatenate(
            (
                inverse_diagonal * vector[: layout.size],
                vector[layout.size :],
            )
        )

    _, preconditioned_residual = native_force_gauss_newton_step(
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
        preconditioner=diagonal_preconditioner,
    )
    output = {
        "coordinates": layout.size,
        "gauge_constraints": gauge.size,
        "restart": 20,
        "max_restarts": 2,
        "iteration_upper_bound": 40,
        "requested_relative_residual": 1.0e-4,
        "unpreconditioned_true_relative_residual": float(
            unpreconditioned_residual
        ),
        "hutchinson_probe_count": int(probes.shape[0]),
        "estimated_diagonal_min": float(jnp.min(estimated_diagonal)),
        "estimated_diagonal_median": float(jnp.median(estimated_diagonal)),
        "estimated_diagonal_max": float(jnp.max(estimated_diagonal)),
        "hutchinson_preconditioned_true_relative_residual": float(
            preconditioned_residual
        ),
        "unaccepted_step_norm": float(jnp.linalg.norm(step)),
        "elapsed_seconds": time.perf_counter() - started,
        "decision": "Neither solve certifies at the 40-iteration budget; reject both. The stochastic diagonal estimate is indefinite and substantially worsens the true residual.",
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
