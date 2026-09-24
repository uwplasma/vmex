"""Bounded nonlinear damped Gauss--Newton polish experiment."""

from __future__ import annotations

import json
import time

import jax.numpy as jnp
import numpy as np

from vmex.core.input import VmecInput
from vmex.core.polish import apply_high_order_correction, make_native_correction_layout
from vmex.core.polish_variational import (
    make_native_gauge_plan,
    make_variational_plan,
    minimum_signed_jacobian,
    native_coordinate_scales,
    native_force_gauss_newton_step,
    native_polish_trial_is_acceptable,
    native_physical_force_residual,
    native_tangential_gauge_residual,
)
from vmex.core.radial_basis import BSplineBasis
from vmex.core.strong_force import certify_strong_force, high_order_state_from_wout


def main() -> None:
    started = time.perf_counter()
    force_scale = 5915447.712414409
    volume_scale = 633.7993467060758
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

    history = []
    linear_tolerance = 1.0e-9
    current_norm = float(jnp.linalg.norm(force_residual(variables[: layout.size])))
    for iteration in range(4):
        step, linear_residual = native_force_gauss_newton_step(
            variables,
            state,
            layout,
            gauge,
            coordinate_scale,
            force_scale,
            volume_scale,
            damping=1.0e-2,
            tolerance=linear_tolerance,
            restart=40,
            max_restarts=40,
        )
        trials = []
        accepted = None
        for fraction in (1.0, 0.5, 0.25, 0.125):
            candidate = variables + fraction * step
            coordinates = candidate[: layout.size]
            corrected = apply_high_order_correction(
                state,
                layout.unpack(coordinate_scale * coordinates),
            )
            candidate_norm = float(jnp.linalg.norm(force_residual(coordinates)))
            candidate_gauge = float(jnp.linalg.norm(gauge_residual(coordinates)))
            minimum_j = float(minimum_signed_jacobian(corrected, plan))
            trial = {
                "fraction": fraction,
                "force_residual_norm": candidate_norm,
                "gauge_residual_norm": candidate_gauge,
                "minimum_signed_jacobian": minimum_j,
                "true_linear_residual": float(linear_residual),
                "requested_linear_tolerance": linear_tolerance,
            }
            trials.append(trial)
            if native_polish_trial_is_acceptable(
                force_before=current_norm,
                force_after=candidate_norm,
                gauge_residual=candidate_gauge,
                minimum_signed_jacobian=minimum_j,
                linear_residual=float(linear_residual),
                linear_tolerance=linear_tolerance,
            ):
                accepted = (candidate, trial)
                break
        history.append(
            {
                "iteration": iteration + 1,
                "initial_force_residual_norm": current_norm,
                "true_linear_residual": float(linear_residual),
                "accepted": accepted is not None,
                "trials": trials,
            }
        )
        if accepted is None:
            break
        variables = accepted[0]
        current_norm = accepted[1]["force_residual_norm"]

    coordinates = variables[: layout.size]
    corrected = apply_high_order_correction(
        state,
        layout.unpack(coordinate_scale * coordinates),
    )
    certificate = certify_strong_force(corrected)
    print(
        json.dumps(
            {
                "coordinates": layout.size,
                "gauge_constraints": gauge.size,
                "curvature_regularization": 0.1,
                "history": history,
                "final_force_residual_norm": current_norm,
                "independent_force_rms_N_per_m3": float(certificate.absolute_l2),
                "independent_epsilon_B": float(
                    certificate.absolute_l2 / force_scale
                ),
                "minimum_signed_jacobian": float(
                    minimum_signed_jacobian(corrected, plan)
                ),
                "independent_radial_refinement_difference": float(
                    certificate.radial_refinement_difference
                ),
                "elapsed_seconds_including_certificate": time.perf_counter()
                - started,
                "decision": "Research-only candidate; each linear solve, gauge residual, nonlinear force descent, Jacobian orientation, and final independent certificate is checked.",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
