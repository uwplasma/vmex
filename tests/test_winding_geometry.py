"""Analytic geometry and AD checks for the winding-surface example helpers."""
import ast
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture(scope="module")
def geometry():
    # Load only pure helpers: importing the example starts a full optimization.
    path = Path(__file__).parents[1] / "examples/optimization/QA_optimization.py"
    names = {
        "_surface_series", "r_surface", "z_surface", "r_surface_prime_phi",
        "z_surface_prime_phi", "r_surface_prime_theta", "z_surface_prime_theta",
        "surface_del_phi", "surface_del_theta", "surface_normal", "surface_unitnormal",
        "surface_coefficients_from_dofs", "points_normals_normal_lengths", "enclosed_volume",
    }
    tree = ast.parse(path.read_text(), filename=str(path))
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in names]
    namespace = {"jax": jax, "jnp": jnp}
    exec(compile(tree, str(path), "exec"), namespace)
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        with jax.disable_jit(False):
            yield namespace
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize("nfp", [1, 2, 5])
@pytest.mark.parametrize("optional", [False, True])
@pytest.mark.parametrize("n_min,m_min", [(None, 0), (-1, 2)])
def test_fourier_values_and_derivatives(geometry, nfp, optional, n_min, m_min):
    rng = np.random.default_rng(17)
    primary, extra = [rng.normal(size=(3, 5)) for _ in range(2)]
    phi, theta = .37, 1.13
    m = np.arange(3)[:, None] + m_min
    n = np.arange(5)[None, :] + (-2 if n_min is None else n_min)
    angle = m * theta - n * nfp * phi
    for family in ("r", "z"):
        c, s = ((primary, extra) if family == "r" else (extra, primary))
        if not optional:
            if family == "r":
                s = np.zeros_like(s)
            else:
                c = np.zeros_like(c)
        expected = np.sum(c * np.cos(angle) + s * np.sin(angle))
        args = (jnp.asarray(primary), nfp, phi, theta,
                jnp.asarray(extra) if optional else None, n_min, m_min)
        value = geometry[f"{family}_surface"](*args)
        np.testing.assert_allclose(value, expected, rtol=2e-13, atol=2e-13)
        for coordinate, frequency in (("phi", -n * nfp), ("theta", m)):
            expected_d = np.sum(frequency * (-c * np.sin(angle) + s * np.cos(angle)))
            actual = geometry[f"{family}_surface_prime_{coordinate}"](*args)
            np.testing.assert_allclose(actual, expected_d, rtol=2e-13, atol=2e-13)
            argnum = 2 if coordinate == "phi" else 3
            derivative = jax.grad(geometry[f"{family}_surface"], argnums=argnum)(*args)
            np.testing.assert_allclose(derivative, actual, rtol=2e-13, atol=2e-13)


def test_broadcast_and_coefficient_ad(geometry):
    coefficients = jnp.arange(15., dtype=jnp.float64).reshape(3, 5) / 15
    phi, theta = jnp.array([.1, .4])[:, None], jnp.array([.2, .7, 1.2])[None, :]
    function = geometry["r_surface"]
    expected = np.array([[function(coefficients, 3, p, t) for t in theta[0]]
                         for p in phi[:, 0]])
    np.testing.assert_allclose(function(coefficients, 3, phi, theta), expected, atol=1e-14)
    def loss(c):
        return jnp.sum(function(c, 3, phi, theta) ** 2)
    direction = jnp.cos(coefficients)
    value, tangent = jax.jit(lambda c, d: jax.jvp(loss, (c,), (d,)))(coefficients, direction)
    reverse = jnp.vdot(jax.jit(jax.grad(loss))(coefficients), direction)
    epsilon = 1e-5
    finite_difference = (loss(coefficients + epsilon * direction)
                         - loss(coefficients - epsilon * direction)) / (2 * epsilon)
    assert np.isfinite(value)
    np.testing.assert_allclose(tangent, reverse, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(tangent, finite_difference, rtol=1e-9)


def test_circular_torus_geometry_and_volume(geometry):
    radius, minor = 3., .7
    # mpol=1, ntor=0: R00, R10, Z10.
    dofs = jnp.array([radius, minor, minor])
    points, normals, jacobian, raw = geometry["points_normals_normal_lengths"](
        dofs, 1, 0, 3, 12, 16)
    phi = np.linspace(0, 2 * np.pi, 12, endpoint=False)[:, None]
    theta = np.linspace(0, 2 * np.pi, 16, endpoint=False)[None, :]
    expected = np.stack(np.broadcast_arrays(np.cos(phi) * np.cos(theta),
                        np.sin(phi) * np.cos(theta), np.sin(theta)), axis=-1)
    np.testing.assert_allclose(normals, expected, atol=3e-15)
    np.testing.assert_allclose(jacobian, np.broadcast_to(
        minor * (radius + minor * np.cos(theta)), (12, 16)), atol=3e-15)
    volume = geometry["enclosed_volume"](points, raw, 12, 16)
    np.testing.assert_allclose(volume, 2 * np.pi**2 * radius * minor**2, rtol=2e-15)

    def volume_of(coefficients):
        position, _, _, normal = geometry["points_normals_normal_lengths"](
            coefficients, 1, 0, 3, 12, 16)
        return geometry["enclosed_volume"](position, normal, 12, 16)

    derivative = jax.jit(jax.grad(volume_of))(dofs)
    np.testing.assert_allclose(derivative, 2 * np.pi**2 * np.array(
        [minor**2, radius * minor, radius * minor]), rtol=2e-15)


def test_array_like_coefficients_and_mismatched_parity(geometry):
    assert geometry["r_surface"]([[3.]], 2, .1, .2) == 3.
    with pytest.raises(ValueError, match="matching shapes"):
        geometry["r_surface"](jnp.ones((2, 3)), 2, .1, .2, jnp.ones((1, 3)))
