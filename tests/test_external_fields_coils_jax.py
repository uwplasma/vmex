from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from vmec_jax._compat import enable_x64
from vmec_jax.external_fields import (
    CoilFieldParams,
    apply_stellarator_symmetry_to_currents,
    apply_stellarator_symmetry_to_curves,
    biot_savart_xyz,
    build_coil_field_geometry,
    coil_coil_distance_soft,
    coil_current_norm,
    coil_curvatures,
    coil_lengths,
    coil_plasma_distance_soft,
    compute_gamma_dash,
    compute_gamma_dashdash,
    curvature_penalty,
    ellipse_coil_field_params,
    ellipse_coil_fourier_dofs,
    fourier_curves_to_gamma,
    length_penalty,
    sample_coil_field_cylindrical,
    sample_coil_field_cylindrical_from_geometry,
    sample_coil_field_cylindrical_from_geometry_jit,
    sample_coil_field_xyz_from_geometry,
    sample_external_field_cylindrical,
)


def _circle_params(*, current: float = 3.0, radius: float = 1.2, n_segments: int = 256) -> CoilFieldParams:
    from vmec_jax._compat import jnp

    dofs = jnp.zeros((1, 3, 3), dtype=float)
    dofs = dofs.at[0, 0, 2].set(radius)  # x = radius * cos(2*pi*t)
    dofs = dofs.at[0, 1, 1].set(radius)  # y = radius * sin(2*pi*t)
    currents = jnp.asarray([current], dtype=float)
    return CoilFieldParams(
        base_curve_dofs=dofs,
        base_currents=currents,
        n_segments=n_segments,
        nfp=1,
        stellsym=False,
    )


def test_fourier_circle_geometry_and_derivatives():
    enable_x64(True)
    params = _circle_params(radius=1.5, n_segments=8)

    gamma = np.asarray(fourier_curves_to_gamma(params.base_curve_dofs, params.n_segments))
    gamma_dash = np.asarray(compute_gamma_dash(params.base_curve_dofs, params.n_segments))
    gamma_dashdash = np.asarray(compute_gamma_dashdash(params.base_curve_dofs, params.n_segments))

    assert gamma.shape == (1, 8, 3)
    np.testing.assert_allclose(gamma[0, 0], [1.5, 0.0, 0.0], atol=1.0e-14)
    np.testing.assert_allclose(gamma[0, 2], [0.0, 1.5, 0.0], atol=1.0e-14)
    np.testing.assert_allclose(gamma_dash[0, 0], [0.0, 2.0 * np.pi * 1.5, 0.0], atol=1.0e-14)
    np.testing.assert_allclose(gamma_dashdash[0, 0], [-(2.0 * np.pi) ** 2 * 1.5, 0.0, 0.0], atol=1.0e-13)


def test_ellipse_coil_helpers_build_oriented_fourier_coils():
    enable_x64(True)
    dofs = np.asarray(
        ellipse_coil_fourier_dofs(
            center=[1.0, 2.0, 3.0],
            normal=[1.0, 0.0, 0.0],
            major_axis=[0.0, 0.0, 1.0],
            major_radius=0.5,
            minor_radius=0.25,
        )
    )
    np.testing.assert_allclose(dofs[:, 0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(dofs[:, 1], [0.0, -0.25, 0.0])
    np.testing.assert_allclose(dofs[:, 2], [0.0, 0.0, 0.5])

    params = ellipse_coil_field_params(
        centers=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        normals=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        major_axes=[0.0, 0.0, 1.0],
        major_radius=0.5,
        minor_radius=[0.25, 0.3],
        currents=2.0,
        n_segments=16,
    )
    assert isinstance(params, CoilFieldParams)
    assert np.asarray(params.base_curve_dofs).shape == (2, 3, 3)
    np.testing.assert_allclose(np.asarray(params.base_currents), [2.0, 2.0])
    gamma = np.asarray(fourier_curves_to_gamma(params.base_curve_dofs, params.n_segments))
    assert gamma.shape == (2, 16, 3)
    np.testing.assert_allclose(gamma[0, 0], [1.0, 0.0, 0.5], atol=1.0e-14)


def test_coil_params_are_pytree_leaves_and_order_zero_geometry_is_static():
    pytest.importorskip("jax")
    from vmec_jax._compat import jnp, tree_util

    params = CoilFieldParams(
        base_curve_dofs=jnp.asarray([[[1.0], [2.0], [3.0]]]),
        base_currents=jnp.asarray([4.0]),
        n_segments=5,
        nfp=2,
        stellsym=True,
        current_scale=3.0,
        regularization_epsilon=1.0e-6,
        chunk_size=4,
    )
    children, treedef = tree_util.tree_flatten(params)
    rebuilt = tree_util.tree_unflatten(treedef, children)

    assert isinstance(rebuilt, CoilFieldParams)
    assert rebuilt.n_segments == 5
    assert rebuilt.nfp == 2
    assert rebuilt.stellsym
    assert rebuilt.current_scale == pytest.approx(3.0)
    assert rebuilt.regularization_epsilon == pytest.approx(1.0e-6)
    assert rebuilt.chunk_size == 4

    gamma = np.asarray(fourier_curves_to_gamma(params.base_curve_dofs, params.n_segments))
    gamma_dash = np.asarray(compute_gamma_dash(params.base_curve_dofs, params.n_segments))
    gamma_dashdash = np.asarray(compute_gamma_dashdash(params.base_curve_dofs, params.n_segments))
    np.testing.assert_allclose(gamma, np.broadcast_to([[[1.0, 2.0, 3.0]]], (1, 5, 3)))
    np.testing.assert_allclose(gamma_dash, 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(gamma_dashdash, 0.0, rtol=0.0, atol=0.0)

    updated = params.with_arrays(base_currents=jnp.asarray([5.0]))
    np.testing.assert_allclose(np.asarray(updated.base_currents), [5.0])
    np.testing.assert_allclose(np.asarray(updated.base_curve_dofs), np.asarray(params.base_curve_dofs))


def test_coil_fourier_geometry_rejects_invalid_dof_shapes():
    bad = np.zeros((1, 2, 3))
    with pytest.raises(ValueError, match=r"base_curve_dofs"):
        fourier_curves_to_gamma(bad, 8)
    with pytest.raises(ValueError, match=r"base_curve_dofs"):
        compute_gamma_dash(bad, 8)
    with pytest.raises(ValueError, match=r"base_curve_dofs"):
        compute_gamma_dashdash(bad, 8)

    even_bad = np.zeros((1, 3, 2))
    with pytest.raises(ValueError, match=r"base_curve_dofs"):
        fourier_curves_to_gamma(even_bad, 8)


def test_circular_coil_on_axis_matches_analytic_biot_savart():
    enable_x64(True)
    params = _circle_params(current=5.0, radius=1.3, n_segments=64)
    gamma = fourier_curves_to_gamma(params.base_curve_dofs, params.n_segments)
    gamma_dash = compute_gamma_dash(params.base_curve_dofs, params.n_segments)
    point = np.asarray([[0.0, 0.0, 0.7]])

    field = np.asarray(biot_savart_xyz(point, gamma, gamma_dash, params.base_currents))
    expected_bz = 1.0e-7 * 5.0 * 2.0 * np.pi * 1.3**2 / (1.3**2 + 0.7**2) ** 1.5

    np.testing.assert_allclose(field[0, 0], 0.0, atol=1.0e-18)
    np.testing.assert_allclose(field[0, 1], 0.0, atol=1.0e-18)
    np.testing.assert_allclose(field[0, 2], expected_bz, rtol=1.0e-13, atol=1.0e-18)


def test_circular_loop_provider_on_axis_matches_analytic_field_profile():
    enable_x64(True)
    current = 4.25
    radius = 0.85
    params = _circle_params(current=current, radius=radius, n_segments=48)
    R = np.zeros(5)
    Z = np.asarray([-1.2, -0.4, 0.0, 0.35, 1.1])
    phi = np.asarray([0.0, 0.3, 1.1, 2.0, 5.2])

    br, bphi, bz = sample_coil_field_cylindrical(params, R, Z, phi)
    expected_bz = 1.0e-7 * current * 2.0 * np.pi * radius**2 / (radius**2 + Z**2) ** 1.5

    np.testing.assert_allclose(br, 0.0, atol=2.0e-18)
    np.testing.assert_allclose(bphi, 0.0, atol=2.0e-18)
    np.testing.assert_allclose(bz, expected_bz, rtol=1.0e-13, atol=1.0e-18)


def test_biot_savart_field_is_linear_in_currents_and_superposes_coils():
    from vmec_jax._compat import jnp

    enable_x64(True)
    dofs = jnp.zeros((2, 3, 3), dtype=float)
    dofs = dofs.at[:, 0, 2].set(0.55)
    dofs = dofs.at[:, 1, 1].set(0.55)
    dofs = dofs.at[0, 2, 0].set(0.35)
    dofs = dofs.at[1, 2, 0].set(-0.45)
    params = CoilFieldParams(
        base_curve_dofs=dofs,
        base_currents=jnp.asarray([2.0, -1.25], dtype=float),
        n_segments=96,
    )
    R = np.asarray([[0.20, 0.45], [0.30, 0.10]])
    Z = np.asarray([[0.10, -0.20], [0.55, -0.60]])
    phi = np.asarray([[0.0, 0.4], [1.1, 2.0]])

    combined = sample_coil_field_cylindrical(params, R, Z, phi)
    first_only = sample_coil_field_cylindrical(params.with_arrays(base_currents=jnp.asarray([2.0, 0.0])), R, Z, phi)
    second_only = sample_coil_field_cylindrical(params.with_arrays(base_currents=jnp.asarray([0.0, -1.25])), R, Z, phi)
    reversed_current = sample_coil_field_cylindrical(
        params.with_arrays(base_currents=-params.base_currents),
        R,
        Z,
        phi,
    )

    for actual, first, second, reversed_component in zip(combined, first_only, second_only, reversed_current, strict=True):
        np.testing.assert_allclose(actual, first + second, rtol=1.0e-14, atol=1.0e-18)
        np.testing.assert_allclose(reversed_component, -actual, rtol=1.0e-14, atol=1.0e-18)


def test_circular_loop_field_has_axisymmetry_and_midplane_parity():
    enable_x64(True)
    params = _circle_params(current=3.0, radius=1.0, n_segments=384)
    R = np.asarray([0.42, 0.42, 0.42, 0.42])
    Z = np.asarray([0.55, 0.55, -0.55, -0.55])
    phi_step = 2.0 * np.pi * 17.0 / params.n_segments
    phi = np.asarray([0.0, phi_step, 0.0, phi_step])

    br, bphi, bz = sample_coil_field_cylindrical(params, R, Z, phi)

    np.testing.assert_allclose(bphi, 0.0, atol=1.0e-17)
    np.testing.assert_allclose(br[0], br[1], rtol=1.0e-13, atol=1.0e-18)
    np.testing.assert_allclose(br[2], br[3], rtol=1.0e-13, atol=1.0e-18)
    np.testing.assert_allclose(bz[0], bz[1], rtol=1.0e-13, atol=1.0e-18)
    np.testing.assert_allclose(bz[2], bz[3], rtol=1.0e-13, atol=1.0e-18)
    np.testing.assert_allclose(br[0], -br[2], rtol=1.0e-13, atol=1.0e-18)
    np.testing.assert_allclose(bz[0], bz[2], rtol=1.0e-13, atol=1.0e-18)


def test_sample_coil_field_cylindrical_shapes_and_dispatch():
    enable_x64(True)
    params = _circle_params(current=2.0, radius=1.0, n_segments=96)
    R = np.asarray([[0.2, 0.4], [0.3, 0.5]])
    Z = np.asarray([[0.1, 0.2], [0.3, 0.4]])
    phi = np.asarray([[0.0, 0.1], [0.2, 0.3]])

    direct = sample_coil_field_cylindrical(params, R, Z, phi)
    dispatch = sample_external_field_cylindrical("direct_coils", None, params, R, Z, phi)
    cached_dispatch = sample_external_field_cylindrical(
        "direct_coils",
        {"coil_geometry": build_coil_field_geometry(params)},
        params,
        R,
        Z,
        phi,
    )

    assert all(component.shape == R.shape for component in direct)
    for actual, expected in zip(dispatch, direct, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1.0e-14, atol=1.0e-18)
    for actual, expected in zip(cached_dispatch, direct, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1.0e-14, atol=1.0e-18)


def test_cached_coil_geometry_sampling_matches_full_sampling():
    enable_x64(True)
    params = _circle_params(current=2.0, radius=1.0, n_segments=96)
    params = replace(params, nfp=2, stellsym=True, regularization_epsilon=1.0e-9, chunk_size=2)
    R = np.asarray([[0.2, 0.4], [0.3, 0.5]])
    Z = np.asarray([[0.1, 0.2], [0.3, 0.4]])
    phi = np.asarray([[0.0, 0.1], [0.2, 0.3]])

    full = sample_coil_field_cylindrical(params, R, Z, phi)
    geometry = build_coil_field_geometry(params)
    cached = sample_coil_field_cylindrical_from_geometry(
        geometry,
        R,
        Z,
        phi,
        regularization_epsilon=params.regularization_epsilon,
        chunk_size=params.chunk_size,
    )

    for actual, expected in zip(cached, full, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1.0e-14, atol=1.0e-18)


def test_jitted_cached_coil_geometry_sampling_matches_full_sampling():
    pytest.importorskip("jax")
    enable_x64(True)
    params = _circle_params(current=2.0, radius=1.0, n_segments=96)
    params = replace(params, nfp=2, stellsym=True, regularization_epsilon=1.0e-9)
    R = np.asarray([[0.2, 0.4], [0.3, 0.5]])
    Z = np.asarray([[0.1, 0.2], [0.3, 0.4]])
    phi = np.asarray([[0.0, 0.1], [0.2, 0.3]])

    full = sample_coil_field_cylindrical(params, R, Z, phi)
    geometry = build_coil_field_geometry(params)
    jitted = sample_coil_field_cylindrical_from_geometry_jit(
        geometry,
        R,
        Z,
        phi,
        regularization_epsilon=params.regularization_epsilon,
    )
    dispatch = sample_external_field_cylindrical(
        "direct_coils",
        {
            "coil_geometry": geometry,
            "regularization_epsilon": params.regularization_epsilon,
            "jit_sampler": True,
        },
        params,
        R,
        Z,
        phi,
    )

    for actual, expected in zip(jitted, full, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1.0e-14, atol=1.0e-18)
    for actual, expected in zip(dispatch, full, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1.0e-14, atol=1.0e-18)


def test_cached_xyz_geometry_sampling_matches_direct_biot_savart_with_chunking():
    enable_x64(True)
    params = replace(
        _circle_params(current=1.7, radius=0.9, n_segments=48),
        nfp=2,
        stellsym=True,
        current_scale=1.25,
        regularization_epsilon=1.0e-8,
    )
    geometry = build_coil_field_geometry(params)
    points = np.asarray(
        [
            [[0.20, 0.10, 0.30], [0.35, -0.20, -0.10]],
            [[-0.25, 0.15, 0.20], [0.10, 0.40, -0.25]],
        ]
    )

    cached_chunked = sample_coil_field_xyz_from_geometry(
        geometry,
        points,
        regularization_epsilon=params.regularization_epsilon,
        chunk_size=3,
    )
    direct_unchunked = biot_savart_xyz(
        points,
        *geometry,
        regularization_epsilon=params.regularization_epsilon,
    )

    np.testing.assert_allclose(cached_chunked, direct_unchunked, rtol=1.0e-14, atol=1.0e-18)


def test_chunked_and_unchunked_biot_savart_match():
    enable_x64(True)
    params = _circle_params(current=2.0, radius=1.0, n_segments=64)
    gamma = fourier_curves_to_gamma(params.base_curve_dofs, params.n_segments)
    gamma_dash = compute_gamma_dash(params.base_curve_dofs, params.n_segments)
    points = np.asarray(
        [
            [0.1, 0.2, 0.3],
            [0.4, -0.1, 0.2],
            [0.2, 0.3, -0.4],
            [-0.3, 0.1, 0.2],
            [0.1, -0.2, -0.3],
        ]
    )

    unchunked = biot_savart_xyz(points, gamma, gamma_dash, params.base_currents)
    chunked = biot_savart_xyz(points, gamma, gamma_dash, params.base_currents, chunk_size=2)

    np.testing.assert_allclose(chunked, unchunked, rtol=1.0e-14, atol=1.0e-18)


def test_chunked_cylindrical_sampling_preserves_current_gradient():
    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp

    enable_x64(True)
    dofs = jnp.zeros((2, 3, 3), dtype=float)
    dofs = dofs.at[:, 0, 2].set(0.75)
    dofs = dofs.at[:, 1, 1].set(0.75)
    dofs = dofs.at[0, 0, 0].set(1.1)
    dofs = dofs.at[1, 0, 0].set(-0.8)
    params = CoilFieldParams(
        base_curve_dofs=dofs,
        base_currents=jnp.asarray([2.0, -1.5], dtype=float),
        n_segments=40,
        regularization_epsilon=1.0e-8,
    )
    R = jnp.asarray([[0.20, 0.35, 0.50], [0.40, 0.30, 0.25]], dtype=float)
    Z = jnp.asarray([[0.15, -0.10, 0.25], [-0.20, 0.05, 0.30]], dtype=float)
    phi = jnp.asarray([[0.0, 0.3, 0.6], [0.9, 1.2, 1.5]], dtype=float)
    weights = jnp.asarray([[0.2, -0.4, 0.6], [-0.8, 1.0, -1.2]], dtype=float)

    def weighted_bz(currents, chunk_size):
        trial = replace(params.with_arrays(base_currents=currents), chunk_size=chunk_size)
        return jnp.sum(weights * sample_coil_field_cylindrical(trial, R, Z, phi)[2])

    currents = params.base_currents
    unchunked_grad = jax.grad(lambda x: weighted_bz(x, None))(currents)
    chunked_grad = jax.grad(lambda x: weighted_bz(x, 4))(currents)

    np.testing.assert_allclose(chunked_grad, unchunked_grad, rtol=1.0e-12, atol=1.0e-18)
    assert np.all(np.isfinite(np.asarray(chunked_grad)))


def test_current_gradient_matches_analytic_linearity():
    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp

    enable_x64(True)
    params = _circle_params(current=4.0, radius=1.1, n_segments=96)

    def bz_for_current(current):
        trial = params.with_arrays(base_currents=jnp.asarray([current]))
        return sample_coil_field_cylindrical(trial, 0.2, 0.3, 0.4)[2]

    value = bz_for_current(4.0)
    grad_value = jax.grad(bz_for_current)(4.0)

    np.testing.assert_allclose(grad_value, value / 4.0, rtol=1.0e-12, atol=1.0e-18)


def test_fourier_coefficient_gradient_matches_finite_difference():
    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp

    enable_x64(True)
    params = _circle_params(current=2.0, radius=1.0, n_segments=128)

    def scalar_for_xcos(xcos):
        dofs = params.base_curve_dofs.at[0, 0, 2].set(xcos)
        trial = params.with_arrays(base_curve_dofs=dofs)
        br, bphi, bz = sample_coil_field_cylindrical(trial, 0.35, 0.25, 0.4)
        return 0.7 * br - 0.2 * bphi + 1.1 * bz

    x0 = jnp.asarray(1.0)
    exact = jax.grad(scalar_for_xcos)(x0)
    eps = 1.0e-5
    fd = (scalar_for_xcos(x0 + eps) - scalar_for_xcos(x0 - eps)) / (2.0 * eps)

    np.testing.assert_allclose(exact, fd, rtol=2.0e-6, atol=1.0e-12)


def test_cached_coil_geometry_sampling_preserves_functional_geometry_gradient():
    pytest.importorskip("jax")
    from vmec_jax._compat import jax

    enable_x64(True)
    params = _circle_params(current=2.0, radius=1.0, n_segments=128)

    def scalar_cached(dofs):
        trial = params.with_arrays(base_curve_dofs=dofs)
        geometry = build_coil_field_geometry(trial)
        br, bphi, bz = sample_coil_field_cylindrical_from_geometry(
            geometry,
            0.35,
            0.25,
            0.4,
            regularization_epsilon=trial.regularization_epsilon,
            chunk_size=trial.chunk_size,
        )
        return 0.7 * br - 0.2 * bphi + 1.1 * bz

    def scalar_full(dofs):
        trial = params.with_arrays(base_curve_dofs=dofs)
        br, bphi, bz = sample_coil_field_cylindrical(trial, 0.35, 0.25, 0.4)
        return 0.7 * br - 0.2 * bphi + 1.1 * bz

    cached_grad = jax.grad(scalar_cached)(params.base_curve_dofs)
    full_grad = jax.grad(scalar_full)(params.base_curve_dofs)

    np.testing.assert_allclose(cached_grad, full_grad, rtol=1.0e-12, atol=1.0e-18)
    assert float(np.linalg.norm(np.asarray(cached_grad))) > 0.0


def test_coordinate_derivative_matches_finite_difference():
    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp

    enable_x64(True)
    params = _circle_params(current=2.0, radius=1.0, n_segments=128)

    def scalar_for_coords(coords):
        R, Z, phi = coords
        br, bphi, bz = sample_coil_field_cylindrical(params, R, Z, phi)
        return 0.5 * br - 0.3 * bphi + 0.9 * bz

    coords = jnp.asarray([0.35, 0.25, 0.4])
    exact = jax.jacfwd(scalar_for_coords)(coords)
    eps = 1.0e-5
    fd_columns = []
    for i in range(3):
        step = np.zeros(3)
        step[i] = eps
        fd_columns.append((scalar_for_coords(coords + step) - scalar_for_coords(coords - step)) / (2.0 * eps))
    fd = jnp.asarray(fd_columns)

    np.testing.assert_allclose(exact, fd, rtol=2.0e-6, atol=1.0e-12)


def test_symmetry_expansion_matches_essos_ordering_for_currents_and_shapes():
    enable_x64(True)
    params = _circle_params(current=3.0, radius=1.0, n_segments=8)
    params = replace(params, nfp=2, stellsym=True)

    curves = apply_stellarator_symmetry_to_curves(params.base_curve_dofs, nfp=params.nfp, stellsym=params.stellsym)
    currents = apply_stellarator_symmetry_to_currents(params.base_currents, nfp=params.nfp, stellsym=params.stellsym)

    assert curves.shape == (4, 3, 3)
    np.testing.assert_allclose(currents, [3.0, -3.0, 3.0, -3.0], rtol=0.0, atol=0.0)


def test_coil_engineering_metrics_are_finite_and_differentiable():
    pytest.importorskip("jax")
    from vmec_jax._compat import jax

    enable_x64(True)
    params = _circle_params(current=2.0, radius=1.0, n_segments=96)
    boundary = np.asarray(
        [
            [0.2, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [-0.2, 0.0, 0.0],
            [0.0, -0.2, 0.0],
        ]
    )

    lengths = np.asarray(coil_lengths(params))
    curvatures = np.asarray(coil_curvatures(params))
    plasma_distance = coil_plasma_distance_soft(params, boundary, alpha=10.0)

    assert lengths.shape == (1,)
    assert curvatures.shape == (1, params.n_segments)
    np.testing.assert_allclose(lengths[0], 2.0 * np.pi, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(curvatures, 1.0, rtol=1.0e-12, atol=1.0e-12)
    assert float(coil_current_norm(params)) == pytest.approx(2.0)
    assert float(plasma_distance) > 0.0
    assert np.isfinite(float(length_penalty(params, maximum=10.0)))
    assert np.isfinite(float(curvature_penalty(params, maximum=2.0)))

    def length_for_radius(radius):
        trial = _circle_params(current=2.0, radius=radius, n_segments=64)
        return coil_lengths(trial)[0]

    np.testing.assert_allclose(jax.grad(length_for_radius)(1.0), 2.0 * np.pi, rtol=1.0e-12, atol=1.0e-12)


def test_coil_coil_distance_soft_uses_distinct_coils_only():
    from vmec_jax._compat import jnp

    enable_x64(True)
    dofs = jnp.zeros((2, 3, 3), dtype=float)
    dofs = dofs.at[:, 0, 2].set(0.2)
    dofs = dofs.at[:, 1, 1].set(0.2)
    dofs = dofs.at[0, 0, 0].set(1.0)
    dofs = dofs.at[1, 0, 0].set(-1.0)
    params = CoilFieldParams(
        base_curve_dofs=dofs,
        base_currents=jnp.asarray([1.0, 1.0]),
        n_segments=24,
    )

    distance = float(coil_coil_distance_soft(params, alpha=20.0))

    assert 1.0 < distance < 2.0
