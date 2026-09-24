"""One bounded gauge-projected force-descent step on the regularized lift.

This is a diagnostic reference, not a production polish solver.  It keeps the
physical-force Jacobian matrix-free (JVP/VJP); only the small gauge Jacobian is
materialized to project the descent direction.
"""

from __future__ import annotations

import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from vmex.core.input import VmecInput
from vmex.core.polish import make_native_correction_layout
from vmex.core.polish_variational import (
    apply_high_order_correction,
    make_native_gauge_plan,
    make_variational_plan,
    minimum_signed_jacobian,
    native_coordinate_scales,
    native_physical_force_residual,
    native_tangential_gauge_residual,
)
from vmex.core.radial_basis import BSplineBasis
from vmex.core.strong_force import certify_strong_force, high_order_state_from_wout


def main() -> None:
    started = time.perf_counter()
    input_path = "examples/data/input.shaped_tokamak_pressure"
    wout_path = "artifacts/p0/run2/wout_shaped_tokamak_pressure.nc"
    force_scale = 5915447.712414409
    volume_scale = 633.7993467060758
    basis = BSplineBasis.clamped(np.linspace(0.0, 1.0, 41), degree=3)
    inp = VmecInput.from_file(input_path)
    state = high_order_state_from_wout(
        wout_path,
        inp=inp,
        radial_basis=basis,
        curvature_regularization=0.1,
    )
    plan = make_variational_plan(state, radial_order=4, ntheta=49, nzeta=1)
    layout = make_native_correction_layout(state)
    gauge = make_native_gauge_plan(state, plan)
    coordinate_scale = native_coordinate_scales(state, layout, plan)
    zero = jnp.zeros((layout.size,), dtype=jnp.float64)

    def force_residual(value):
        return native_physical_force_residual(
            value,
            state,
            layout,
            gauge,
            coordinate_scale,
            force_scale,
            volume_scale,
        )

    def gauge_residual(value):
        return native_tangential_gauge_residual(
            state,
            layout.unpack(coordinate_scale * value),
            gauge,
        )

    gauge_matrix = np.asarray(jax.jacfwd(gauge_residual)(zero))

    def project(gradient):
        multiplier = np.linalg.lstsq(
            gauge_matrix @ gauge_matrix.T,
            gauge_matrix @ np.asarray(gradient),
            rcond=1.0e-12,
        )[0]
        return gradient - gauge_matrix.T @ multiplier

    objective_gradient = jax.grad(
        lambda value: 0.5 * jnp.vdot(force_residual(value), force_residual(value))
    )
    coordinates = zero
    history = []
    for iteration in range(5):
        residual = force_residual(coordinates)
        gradient = objective_gradient(coordinates)
        projected_gradient = project(gradient)
        direction = -projected_gradient
        directional_force = jax.jvp(
            force_residual, (coordinates,), (jnp.asarray(direction),)
        )[1]
        step = float(
            jnp.vdot(projected_gradient, projected_gradient)
            / jnp.vdot(directional_force, directional_force)
        )
        accepted = None
        trials = []
        for fraction in (1.0, 0.5, 0.25, 0.125):
            candidate = coordinates + fraction * step * jnp.asarray(direction)
            force = force_residual(candidate)
            corrected = apply_high_order_correction(
                state,
                layout.unpack(coordinate_scale * candidate),
            )
            minimum_j = float(minimum_signed_jacobian(corrected, plan))
            candidate_norm = float(jnp.linalg.norm(force))
            result = {
                "step_fraction": fraction,
                "force_residual_norm": candidate_norm,
                "minimum_signed_jacobian": minimum_j,
                "gauge_residual_norm": float(jnp.linalg.norm(gauge_residual(candidate))),
            }
            trials.append(result)
            if candidate_norm < float(jnp.linalg.norm(residual)) and minimum_j > 0.0:
                accepted = (candidate, result)
                break
        history.append(
            {
                "iteration": iteration + 1,
                "initial_force_residual_norm": float(jnp.linalg.norm(residual)),
                "projected_gradient_norm": float(np.linalg.norm(projected_gradient)),
                "linearized_step_length": step,
                "trials": trials,
                "accepted": accepted is not None,
            }
        )
        if accepted is None:
            break
        coordinates = accepted[0]

    selected = apply_high_order_correction(
        state,
        layout.unpack(coordinate_scale * coordinates),
    )
    certificate = certify_strong_force(selected)
    output = {
        "basis_size": basis.size,
        "curvature_regularization": 0.1,
        "coordinates": layout.size,
        "gauge_constraints": gauge.size,
        "materialized_gauge_jacobian_bytes": int(gauge_matrix.nbytes),
        "force_scale_N_per_m3": force_scale,
        "volume_scale_m3": volume_scale,
        "initial_force_residual_norm": history[0]["initial_force_residual_norm"],
        "iterations": history,
        "independent_force_rms_N_per_m3": float(certificate.absolute_l2),
        "independent_epsilon_B": float(certificate.absolute_l2 / force_scale),
        "independent_radial_refinement_difference": float(
            certificate.radial_refinement_difference
        ),
        "elapsed_seconds_including_certificate": time.perf_counter() - started,
        "limitation": "at most five projected-gradient steps; gauge Jacobian materialized; no convergence claim",
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
