"""Analytic geometry and AD checks for the winding-surface example helpers."""
import ast
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture(scope="module")
def geometry():
    from jaxopt import LBFGS
    from jaxopt.implicit_diff import root_jvp
    from solvax import gcrot

    # Load only pure helpers: importing the example starts a full optimization.
    path = Path(__file__).parents[1] / "examples/optimization/QA_optimization.py"
    names = {
        "_surface_series", "r_surface", "z_surface", "r_surface_prime_phi",
        "z_surface_prime_phi", "r_surface_prime_theta", "z_surface_prime_theta",
        "surface_del_phi", "surface_del_theta", "surface_normal", "surface_unitnormal",
        "surface_coefficients_from_dofs", "points_normals_normal_lengths", "enclosed_volume",
        "_ntor_from_coefficients", "reduced_memory_induction_matrix", "spectral_width",
        "smooth_minimum_distance", "calc_objectives", "_calc_objectives",
        "_periodic_objectives", "_make_winding_objective",
        "_winding_linear_solve", "_make_winding_solver", "winding_surface_objective",
        "coefficients_to_dofs", "_validate_matching_shape", "mode_weights_from_coefficients",
    }
    tree = ast.parse(path.read_text(), filename=str(path))
    aliases = {"_winding_objective_value", "_periodic_objective_value",
               "_solve_winding_surface", "_solve_periodic_winding"}
    tree.body = [node for node in tree.body
                 if (isinstance(node, ast.FunctionDef) and node.name in names)
                 or (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                     and node.targets[0].id in aliases)]
    namespace = {
        "jax": jax, "jnp": jnp, "minimize": LBFGS, "root_jvp": root_jvp, "gcrot": gcrot, "MU0": 4 * np.pi * 1e-7, "nfp": 2,
        "WINDING_NPHI": 12, "WINDING_NTHETA": 12, "MINIMUM_DISTANCE": .05,
        "DISTANCE_WALL_SCALE": .01, "DISTANCE_WALL_WEIGHT": 10.,
        "PCA_WEIGHT": 1., "VOLUME_WEIGHT": .02, "SPECTRAL_WEIGHT": .02,
    }
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


def test_winding_objective_hvp_matches_gradient_difference(geometry):
    # Break repeated singular values without altering the entropy definition.
    rc, zs = np.zeros((3, 5)), np.zeros((3, 5))
    rc[0, 2], rc[1, 2], zs[1, 2] = 3., .7, .7
    dofs = np.r_[rc.ravel()[2:], zs.ravel()[3:]]
    dofs = jnp.asarray(dofs + .003 * np.random.default_rng(12).normal(size=dofs.shape))
    plasma = (.75 * dofs).at[0].set(dofs[0])
    points, normals, _, _ = geometry["points_normals_normal_lengths"](
        plasma, 2, 2, 2, 12, 12)
    points, normals = points.reshape(-1, 3), normals.reshape(-1, 3)
    weights = jnp.ones_like(dofs)
    scales = geometry["calc_objectives"](dofs, points, normals, weights, rc)

    def objective(value):
        return geometry["_winding_objective_value"](value, points, normals, weights, rc, scales)

    direction = jnp.sin(dofs + .1)
    gradient = jax.jit(jax.grad(objective))
    hvp = jax.jit(lambda value: jax.jvp(
        jax.grad(objective), (value,), (direction,))[1])(dofs)
    assert np.all(np.isfinite(hvp))
    for epsilon in (1e-5, 1e-6):
        finite_difference = (gradient(dofs + epsilon * direction)
                             - gradient(dofs - epsilon * direction)) / (2 * epsilon)
        np.testing.assert_allclose(hvp, finite_difference, rtol=2e-5, atol=2e-7)


def test_float32_geometry_keeps_angular_precision(geometry):
    rc = jnp.array([[3.], [.7]], dtype=jnp.float32)
    phi, theta = jnp.asarray(.1, dtype=jnp.float32), jnp.asarray(.3, dtype=jnp.float32)
    value = geometry["r_surface"](rc, 2, phi, theta)
    assert value.dtype == jnp.float32
    np.testing.assert_allclose(value, 3 + .7 * np.cos(.3), rtol=2e-7)


@pytest.fixture(scope="module")
def response(geometry):
    return geometry


@pytest.mark.parametrize("matrix", [
    [[3., 1.], [1., -2.]], [[3., 2.], [-1., 4.]],
])
def test_certified_winding_linear_solve_forward_reverse(response, matrix):
    matrix = jnp.asarray(matrix)
    rhs = jnp.array([.7, -.3])
    def solve(value):
        return response["_winding_linear_solve"](lambda x: matrix @ x, value)
    expected = jnp.linalg.solve(matrix, rhs)
    np.testing.assert_allclose(jax.jit(solve)(rhs), expected, rtol=1e-12)
    direction = jnp.array([.2, .8])
    tangent = jax.jvp(solve, (rhs,), (direction,))[1]
    cotangent = jax.grad(lambda value: jnp.dot(solve(value), direction))(rhs)
    np.testing.assert_allclose(tangent, jnp.linalg.solve(matrix, direction), rtol=1e-12)
    np.testing.assert_allclose(cotangent, jnp.linalg.solve(matrix.T, direction), rtol=1e-12)


def test_winding_linear_failure_is_not_an_unchecked_gradient(response):
    zero = response["_winding_linear_solve"](lambda x: 2 * x, jnp.zeros(2))
    np.testing.assert_array_equal(zero, jnp.zeros(2))
    value = response["_winding_linear_solve"](
        lambda x: 2 * x, jnp.ones(2), max_restarts=0)
    assert np.all(np.isnan(value))


def test_winding_root_response_and_curvature(response):
    matrix = jnp.array([[3., .4], [.4, 2.]])
    solver = response["_make_winding_solver"](
        lambda d, p, ref, weights, rc, scales: .5 * d @ (matrix + jnp.diag(p**2 + scales**2)) @ d
        - jnp.dot(p**2, d))
    def solve(p):
        return solver(jnp.zeros(2), p, 0., 0., 0., .1 * p)
    p = jnp.array([.7, -.3])
    direction = jnp.array([.2, .8])
    value, tangent = jax.jvp(solve, (p,), (direction,))
    def exact(q):
        return jnp.linalg.solve(matrix + jnp.diag(1.01 * q**2), q**2)
    np.testing.assert_allclose(value, exact(p), atol=5e-7)
    np.testing.assert_allclose(tangent, jax.jvp(exact, (p,), (direction,))[1], atol=2e-7)
    gradient = jax.grad(lambda q: jnp.sum(solve(q)))
    exact_gradient = jax.grad(lambda q: jnp.sum(exact(q)))
    np.testing.assert_allclose(gradient(p), exact_gradient(p), atol=2e-7)
    hvp = jax.jvp(gradient, (p,), (direction,))[1]
    np.testing.assert_allclose(hvp, jax.jvp(exact_gradient, (p,), (direction,))[1], atol=2e-7)


def test_unstationary_winding_inner_solve_rejects_response(response):
    solver = response["_make_winding_solver"](lambda d, p, *unused: jnp.dot(d, p))
    _, tangent = jax.jvp(
        lambda p: solver(jnp.zeros(2), p, 0., 0., 0., 0.),
        (jnp.ones(2),), (jnp.ones(2),))
    assert np.all(np.isnan(tangent))


@pytest.mark.parametrize("nfp,nphi", [(2, 12), (1, 12), (3, 12), (2, 11)])
def test_periodic_objective_mixed_coefficient_hvp(geometry, monkeypatch, nfp, nphi):
    monkeypatch.setitem(geometry, "nfp", nfp)
    monkeypatch.setitem(geometry, "WINDING_NPHI", nphi)
    rc, zs = np.zeros((3, 5)), np.zeros((3, 5))
    rc[0, 2], rc[1, 2], zs[1, 2] = 3., .7, .7
    dofs = np.r_[rc.ravel()[2:], zs.ravel()[3:]]
    dofs = jnp.asarray(dofs + .003 * np.random.default_rng(12).normal(size=dofs.shape))
    plasma = (.75 * dofs).at[0].set(dofs[0])
    values = jnp.concatenate((dofs, plasma))
    direction = jnp.asarray(np.random.default_rng(81).normal(size=values.shape))
    direction /= jnp.linalg.norm(direction)
    weights, scales = jnp.ones_like(dofs), jnp.ones(5)
    structured = geometry["_make_winding_objective"](geometry["_periodic_objectives"])

    def objective(value, periodic):
        winding, boundary = jnp.split(value, 2)
        if periodic:
            return structured(winding, boundary, rc, weights, rc, scales)
        points, normals, _, _ = geometry["points_normals_normal_lengths"](
            boundary, 2, 2, nfp, nphi, 12)
        return geometry["_winding_objective_value"](
            winding, points.reshape(-1, 3), normals.reshape(-1, 3), weights, rc, scales)

    def evaluate(periodic):
        def fun(value):
            return objective(value, periodic)
        gradient = jax.grad(fun)
        return jax.jit(lambda value: (fun(value), gradient(value),
            jax.jvp(gradient, (value,), (direction,))[1]))(values)
    full, reduced = evaluate(False), evaluate(True)
    for expected, actual in zip(full, reduced):
        assert np.all(np.isfinite(actual))
        np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=2e-8)
    gradient = jax.jit(jax.grad(lambda value: objective(value, True)))
    h = 1e-6
    finite_difference = (gradient(values + h * direction)
                         - gradient(values - h * direction)) / (2 * h)
    np.testing.assert_allclose(reduced[2], finite_difference, rtol=2e-5, atol=2e-7)


def test_raw_array_entropy_retains_symmetry_breaking_derivatives(geometry):
    rng = np.random.default_rng(17)
    rc, zs = np.zeros((3, 5)), np.zeros((3, 5))
    rc[0, 2], rc[1, 2], zs[1, 2] = 3., .7, .7
    dofs = np.r_[rc.ravel()[2:], zs.ravel()[3:]]
    dofs = jnp.asarray(dofs + .003 * rng.normal(size=dofs.shape))
    plasma = (.75 * dofs).at[0].set(dofs[0])
    points = geometry["points_normals_normal_lengths"]
    wp, wn, _, _ = points(dofs, 2, 2, 2, 12, 12)
    pp, pn, _, _ = points(plasma, 2, 2, 2, 12, 12)
    value = jnp.stack((pp.reshape(-1, 3), pn.reshape(-1, 3)))
    direction = jnp.asarray(rng.normal(size=value.shape))
    direction /= jnp.linalg.norm(direction)

    def full_oracle(raw):
        matrix = geometry["reduced_memory_induction_matrix"](
            wp.reshape(-1, 3), raw[0], wn.reshape(-1, 3), raw[1])
        spectrum = jnp.linalg.svd(matrix, compute_uv=False)
        probabilities = spectrum / jnp.sum(spectrum)
        return -1 / jnp.sum(probabilities * jnp.log(probabilities))

    def generic(raw):
        return geometry["calc_objectives"](dofs, raw[0], raw[1], jnp.ones_like(dofs), rc)[0]

    def projected(raw):
        return geometry["_calc_objectives"](
            dofs, raw[0], raw[1], jnp.ones_like(dofs), rc, periodic=True)[0]

    def evaluate(fun):
        return jax.jit(lambda raw: (fun(raw), jax.grad(fun)(raw),
            jax.jvp(jax.grad(fun), (raw,), (direction,))[1]))(value)
    actual, expected, reduced = evaluate(generic), evaluate(full_oracle), evaluate(projected)
    for result, oracle in zip(actual, expected):
        np.testing.assert_allclose(result, oracle, rtol=2e-8, atol=2e-10)
    np.testing.assert_allclose(actual[0], reduced[0], atol=1e-14)
    # Equal values at symmetry do not authorize restricting arbitrary raw-array AD.
    assert jnp.linalg.norm(actual[2] - reduced[2]) > .01 * jnp.linalg.norm(actual[2])
    h = 1e-6
    gradient = jax.jit(jax.grad(generic))
    finite_difference = (gradient(value + h * direction)
                         - gradient(value - h * direction)) / (2 * h)
    np.testing.assert_allclose(actual[2], finite_difference, rtol=2e-5, atol=1e-8)


def test_equilibrium_winding_routes_physical_coefficients(geometry, monkeypatch):
    rc = jnp.array([[0., 3., .002], [.001, .7, .003]])
    zs = jnp.array([[0., 0., .001], [.001, .7, .002]])
    winding_rc, winding_zs = rc.at[1, 1].set(.9), zs.at[1, 1].set(.9)
    zeros = jnp.zeros_like(rc)
    monkeypatch.setitem(geometry, "_aspect_scalars", lambda *args: (.2,))
    monkeypatch.setitem(geometry, "_geometry", lambda *args: ((rc, zeros, zeros, zs),))
    monkeypatch.setitem(geometry, "vmex_boundary_to_dense", lambda *args: (rc, zeros, zeros, zs))
    monkeypatch.setitem(geometry, "extend_via_normal_jax",
                        lambda *args, **kwargs: (winding_rc, winding_zs, zeros, zeros))
    plasma = geometry["coefficients_to_dofs"](rc, zs)
    winding = geometry["coefficients_to_dofs"](winding_rc, winding_zs)
    captured = []

    def solve(initial, boundary, reference, weights, winding_reference, scales):
        np.testing.assert_array_equal(boundary, plasma)
        np.testing.assert_array_equal(reference, rc)
        np.testing.assert_array_equal(initial, winding)
        captured.append(scales)
        return initial

    monkeypatch.setitem(geometry, "_solve_periodic_winding", solve)
    actual = geometry["winding_surface_objective"](None, None)
    weights = geometry["mode_weights_from_coefficients"](winding_rc)
    expected = geometry["_periodic_objectives"](winding, plasma, rc, weights, winding_rc)
    assert len(captured) == 1
    np.testing.assert_allclose(captured[0], expected)
    np.testing.assert_allclose(actual, 1 + jnp.tanh(1 - expected[3]))
