from __future__ import annotations

import numpy as np
import pytest

from vmec_jax._compat import enable_x64
from vmec_jax.external_fields import CoilFieldParams, coil_curvatures, coil_lengths, sample_coil_field_cylindrical
from vmec_jax.robust_coils import (
    CoilPerturbationSample,
    aggregate_risk,
    displace_curve_dofs,
    identity_coil_perturbation,
    perturb_coil_params,
    rotate_curve_dofs_about_z,
    sample_coil_perturbation,
    sample_coil_perturbations,
)


def _circle_params(*, current: float = 3.0, radius: float = 1.2) -> CoilFieldParams:
    from vmec_jax._compat import jnp

    dofs = jnp.zeros((1, 3, 5), dtype=float)
    dofs = dofs.at[0, 0, 2].set(radius)  # x = radius * cos(2*pi*t)
    dofs = dofs.at[0, 1, 1].set(radius)  # y = radius * sin(2*pi*t)
    dofs = dofs.at[0, 2, 4].set(0.1)  # small z harmonic keeps higher modes present
    currents = jnp.asarray([current], dtype=float)
    return CoilFieldParams(
        base_curve_dofs=dofs,
        base_currents=currents,
        n_segments=16,
        nfp=1,
        stellsym=False,
    )


def test_identity_perturbation_preserves_shapes_and_does_not_mutate_inputs():
    enable_x64(True)
    params = _circle_params(current=4.0, radius=1.5)
    base_dofs = np.asarray(params.base_curve_dofs).copy()
    base_currents = np.asarray(params.base_currents).copy()

    perturbed = perturb_coil_params(params, identity_coil_perturbation(params))

    assert perturbed.base_curve_dofs.shape == params.base_curve_dofs.shape
    assert perturbed.base_currents.shape == params.base_currents.shape
    np.testing.assert_allclose(perturbed.base_curve_dofs, base_dofs, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(perturbed.base_currents, base_currents, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(params.base_curve_dofs, base_dofs, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(params.base_currents, base_currents, rtol=0.0, atol=0.0)


def test_current_perturbation_is_deterministic_for_fixed_prng_key():
    pytest.importorskip("jax")
    from vmec_jax._compat import jax

    enable_x64(True)
    params = _circle_params(current=5.0)
    key = jax.random.PRNGKey(17)

    sample_a = sample_coil_perturbation(key, params, current_sigma=0.15)
    sample_b = sample_coil_perturbation(key, params, current_sigma=0.15)
    perturbed_a = perturb_coil_params(params, sample_a)
    perturbed_b = perturb_coil_params(params, sample_b)

    np.testing.assert_allclose(sample_a.current_factors, sample_b.current_factors, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(perturbed_a.base_currents, perturbed_b.base_currents, rtol=0.0, atol=0.0)
    assert perturbed_a.base_currents.shape == params.base_currents.shape
    assert not np.allclose(perturbed_a.base_currents, params.base_currents)


def test_rigid_displacement_only_changes_constant_fourier_terms():
    enable_x64(True)
    from vmec_jax._compat import jnp

    params = _circle_params()
    displacement = jnp.asarray([0.2, -0.3, 0.4])

    shifted = displace_curve_dofs(params.base_curve_dofs, displacement)

    np.testing.assert_allclose(shifted[:, :, 0], params.base_curve_dofs[:, :, 0] + displacement[None, :])
    np.testing.assert_allclose(shifted[:, :, 1:], params.base_curve_dofs[:, :, 1:])


def test_toroidal_phase_rotation_rotates_all_centerline_dof_vectors_about_z():
    enable_x64(True)
    from vmec_jax._compat import jnp

    params = _circle_params(radius=1.0)

    rotated = rotate_curve_dofs_about_z(params.base_curve_dofs, 0.5 * jnp.pi)

    np.testing.assert_allclose(rotated[0, 0, 1], -1.0, atol=1.0e-15)
    np.testing.assert_allclose(rotated[0, 0, 2], 0.0, atol=1.0e-15)
    np.testing.assert_allclose(rotated[0, 1, 1], 0.0, atol=1.0e-15)
    np.testing.assert_allclose(rotated[0, 1, 2], 1.0, atol=1.0e-15)
    np.testing.assert_allclose(rotated[0, 2, :], params.base_curve_dofs[0, 2, :], atol=1.0e-15)


def test_rigid_centerline_motion_preserves_length_and_curvature():
    enable_x64(True)
    from vmec_jax._compat import jnp

    params = _circle_params(radius=1.1)
    displacement = jnp.asarray([0.35, -0.25, 0.15], dtype=float)
    phase = jnp.asarray(0.37 * jnp.pi, dtype=float)

    translated = params.with_arrays(base_curve_dofs=displace_curve_dofs(params.base_curve_dofs, displacement))
    rotated = params.with_arrays(base_curve_dofs=rotate_curve_dofs_about_z(params.base_curve_dofs, phase))

    np.testing.assert_allclose(coil_lengths(translated), coil_lengths(params), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(coil_curvatures(translated), coil_curvatures(params), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(coil_lengths(rotated), coil_lengths(params), rtol=1.0e-15, atol=1.0e-15)
    np.testing.assert_allclose(coil_curvatures(rotated), coil_curvatures(params), rtol=1.0e-14, atol=1.0e-14)


def test_current_perturbation_scales_direct_biot_savart_field():
    pytest.importorskip("jax")
    from vmec_jax._compat import jnp

    enable_x64(True)
    params = _circle_params(current=4.2, radius=1.15)
    identity = identity_coil_perturbation(params)
    current_factor = 1.75
    sample = CoilPerturbationSample(
        current_factors=jnp.asarray([current_factor], dtype=float),
        displacement_xyz=identity.displacement_xyz,
        toroidal_phase=identity.toroidal_phase,
        centerline_dof_delta=identity.centerline_dof_delta,
    )
    perturbed = perturb_coil_params(params, sample)
    R = jnp.asarray([0.42, 0.74, 1.68], dtype=float)
    Z = jnp.asarray([0.18, -0.24, 0.37], dtype=float)
    phi = jnp.asarray([0.13, 0.91, 2.04], dtype=float)

    base_field = jnp.stack(sample_coil_field_cylindrical(params, R, Z, phi))
    scaled_field = jnp.stack(sample_coil_field_cylindrical(perturbed, R, Z, phi))

    np.testing.assert_allclose(scaled_field, current_factor * base_field, rtol=1.0e-14, atol=1.0e-20)


def test_centerline_gaussian_perturbation_is_deterministic_and_preserves_constants_by_default():
    pytest.importorskip("jax")
    from vmec_jax._compat import jax

    enable_x64(True)
    params = _circle_params()
    key = jax.random.PRNGKey(23)

    sample_a = sample_coil_perturbation(key, params, centerline_sigma=1.0e-3)
    sample_b = sample_coil_perturbation(key, params, centerline_sigma=1.0e-3)

    np.testing.assert_allclose(sample_a.centerline_dof_delta, sample_b.centerline_dof_delta, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(sample_a.centerline_dof_delta[:, :, 0], 0.0, rtol=0.0, atol=0.0)
    assert np.linalg.norm(np.asarray(sample_a.centerline_dof_delta[:, :, 1:])) > 0.0


def test_centerline_gaussian_perturbation_can_include_constant_terms():
    pytest.importorskip("jax")
    from vmec_jax._compat import jax

    enable_x64(True)
    params = _circle_params()
    sample = sample_coil_perturbation(
        jax.random.PRNGKey(29),
        params,
        centerline_sigma=1.0e-3,
        centerline_include_constant=True,
    )

    assert np.linalg.norm(np.asarray(sample.centerline_dof_delta[:, :, 0])) > 0.0


def test_batched_samples_are_vmap_compatible():
    pytest.importorskip("jax")
    from vmec_jax._compat import jax

    enable_x64(True)
    params = _circle_params(current=2.0)
    samples = sample_coil_perturbations(
        jax.random.PRNGKey(31),
        params,
        4,
        current_sigma=0.05,
        displacement_sigma=1.0e-3,
        toroidal_phase_sigma=1.0e-3,
        centerline_sigma=1.0e-4,
    )

    def current_after_perturbation(sample):
        return perturb_coil_params(params, sample).base_currents

    batched_currents = jax.vmap(current_after_perturbation)(samples)

    assert samples.current_factors.shape == (4, 1)
    assert samples.displacement_xyz.shape == (4, 3)
    assert samples.toroidal_phase.shape == (4,)
    assert samples.centerline_dof_delta.shape == (4,) + params.base_curve_dofs.shape
    assert batched_currents.shape == (4, 1)


def test_perturbation_gradients_are_available_for_current_and_geometry_controls():
    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp

    enable_x64(True)
    params = _circle_params(current=3.0)
    identity = identity_coil_perturbation(params)

    def current_objective(factor):
        sample = CoilPerturbationSample(
            current_factors=jnp.asarray([factor]),
            displacement_xyz=identity.displacement_xyz,
            toroidal_phase=identity.toroidal_phase,
            centerline_dof_delta=identity.centerline_dof_delta,
        )
        currents = perturb_coil_params(params, sample).base_currents
        return jnp.sum(currents**2)

    def displacement_objective(dx):
        sample = CoilPerturbationSample(
            current_factors=identity.current_factors,
            displacement_xyz=jnp.asarray([dx, 0.0, 0.0]),
            toroidal_phase=identity.toroidal_phase,
            centerline_dof_delta=identity.centerline_dof_delta,
        )
        dofs = perturb_coil_params(params, sample).base_curve_dofs
        return dofs[0, 0, 0]

    np.testing.assert_allclose(jax.grad(current_objective)(2.0), 36.0, rtol=1.0e-14, atol=1.0e-14)
    np.testing.assert_allclose(jax.grad(displacement_objective)(0.25), 1.0, rtol=1.0e-14, atol=1.0e-14)


def test_aggregate_risk_methods_are_deterministic_shape_preserving_and_differentiable():
    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp

    enable_x64(True)
    values = jnp.asarray([1.0, 2.0, 5.0])
    matrix = jnp.asarray([[1.0, 2.0], [3.0, 6.0], [5.0, 7.0]])

    np.testing.assert_allclose(aggregate_risk(values, "mean"), np.mean([1.0, 2.0, 5.0]))
    expected_mean_plus_std = np.mean([1.0, 2.0, 5.0]) + np.std([1.0, 2.0, 5.0])
    np.testing.assert_allclose(aggregate_risk(values, "mean_plus_std"), expected_mean_plus_std)
    assert float(aggregate_risk(values, "smooth_max", temperature=0.05)) > 4.99
    np.testing.assert_allclose(
        aggregate_risk(matrix, "mean", axis=0),
        np.asarray([3.0, 5.0]),
    )
    np.testing.assert_allclose(
        aggregate_risk(values, "soft_cvar", temperature=0.2, tail_fraction=0.5),
        aggregate_risk(values, "smooth_tail", temperature=0.2, tail_fraction=0.5),
    )
    assert float(aggregate_risk(values, "soft_cvar", temperature=0.2, tail_fraction=0.5)) > float(
        aggregate_risk(values, "mean")
    )

    grad_value = jax.grad(lambda x: aggregate_risk(x, "soft_cvar", temperature=0.2, tail_fraction=0.5))(values)
    assert grad_value.shape == values.shape
    assert np.all(np.isfinite(np.asarray(grad_value)))


def test_robust_sampling_and_risk_inputs_are_validated():
    pytest.importorskip("jax")
    from vmec_jax._compat import jax

    enable_x64(True)
    params = _circle_params()

    with pytest.raises(ValueError, match="n_samples"):
        sample_coil_perturbations(jax.random.PRNGKey(37), params, -1)
    with pytest.raises(ValueError, match="temperature"):
        aggregate_risk([1.0, 2.0], "smooth_max", temperature=0.0)
    with pytest.raises(ValueError, match="tail_fraction"):
        aggregate_risk([1.0, 2.0], "soft_cvar", tail_fraction=0.0)


def test_smooth_max_risk_axis_gradient_matches_softmax_weights():
    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp

    enable_x64(True)
    values = jnp.asarray([[1.0, 2.0, 3.0], [0.5, 4.0, 6.0]], dtype=float)
    row_weights = jnp.asarray([0.25, -0.75], dtype=float)
    temperature = 0.7

    def weighted_row_risk(x):
        return jnp.dot(row_weights, aggregate_risk(x, "smooth_max", axis=1, temperature=temperature))

    grad_value = jax.grad(weighted_row_risk)(values)
    expected = row_weights[:, None] * jax.nn.softmax(values / temperature, axis=1)

    np.testing.assert_allclose(grad_value, expected, rtol=1.0e-12, atol=1.0e-14)
    assert np.all(np.isfinite(np.asarray(grad_value)))


def test_aggregate_risk_rejects_unknown_method():
    with pytest.raises(ValueError, match="method must be one of"):
        aggregate_risk([1.0, 2.0], "unknown")
