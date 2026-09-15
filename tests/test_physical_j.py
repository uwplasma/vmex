"""Small analytic checks for the resolved physical-J backport."""

import jax
import jax.numpy as jnp

from vmex.core.bounce import trace_boozer_field_lines
from vmex.core.maxj import common_trapped_pitches, maximum_j_residual_from_boozer
from vmex.core.qi import j_invariant_qi_residual_from_boozer


jax.config.update("jax_enable_x64", True)


def _physical_arguments(amplitudes=(0.2, 0.2)):
    bmnc = jnp.asarray([[2.0, amplitudes[0]], [2.0, amplitudes[1]]])
    xm = jnp.asarray([0.0, 0.0])
    xn = jnp.asarray([0.0, 2.0])
    iota = jnp.asarray([0.4, 0.5])
    G = jnp.asarray([3.0, 3.0])
    I = jnp.asarray([0.0, 0.0])
    alpha = jnp.linspace(0.0, 2.0 * jnp.pi, 3, endpoint=False)
    trace = trace_boozer_field_lines(
        bmnc_b=bmnc,
        xm_b=xm,
        xn_b=xn,
        iota_b=iota,
        G_b=G,
        I_b=I,
        nfp=2,
        alpha=alpha,
        points_per_period=16,
        num_periods=2,
    )
    pitch = common_trapped_pitches(
        jnp.swapaxes(trace["bmag"], 1, 2), (0.5,)
    )
    return dict(
        bmnc_b=bmnc,
        xm_b=xm,
        xn_b=xn,
        iota_b=iota,
        G_b=G,
        I_b=I,
        nfp=2,
        pitch=pitch,
        nalpha=3,
        points_per_period=16,
        num_periods=2,
        max_wells=4,
        quadrature_order=8,
    )


def test_identical_actual_wells_are_qi_and_radially_flat():
    arguments = _physical_arguments()
    qi = j_invariant_qi_residual_from_boozer(**arguments)
    maxj = maximum_j_residual_from_boozer(
        **arguments,
        psi_b=jnp.asarray([0.25, 0.75]),
        psi_edge=1.0,
    )

    assert jnp.all(jnp.isfinite(qi["residuals1d"]))
    assert jnp.all(jnp.isfinite(maxj["residuals1d"]))
    assert jnp.allclose(qi["total"], 0.0, atol=1.0e-28)
    assert jnp.allclose(maxj["total"], 0.0, atol=1.0e-28)


def test_outward_increasing_action_is_penalized_with_finite_gradient():
    def objective(outer_amplitude):
        arguments = _physical_arguments((0.2, outer_amplitude))
        return maximum_j_residual_from_boozer(
            **arguments,
            psi_b=jnp.asarray([0.25, 0.75]),
            psi_edge=1.0,
        )["total"]

    value, derivative = jax.value_and_grad(objective)(jnp.asarray(0.21))

    assert value > 0.0
    assert jnp.isfinite(derivative)
