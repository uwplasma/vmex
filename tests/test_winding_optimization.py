"""Correctness checks for winding-surface spectral and implicit derivatives."""

import ast
import functools
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxopt import LBFGSB
from jaxopt.implicit_diff import root_jvp
from solvax import gcrot


@pytest.fixture(scope="module")
def winding_helpers():
    """Load pure helpers without executing the optimization example."""
    path = (Path(__file__).parents[1] / "examples" / "optimization"
            / "QA_winding_optimization.py")
    names = {
        "_mode_angle", "r_surface", "z_surface", "r_surface_prime_phi",
        "z_surface_prime_phi", "r_surface_prime_theta",
        "z_surface_prime_theta", "surface_del_phi", "surface_del_theta",
        "surface_normal", "_ntor_from_coefficients",
        "surface_coefficients_from_dofs", "points_normals_normal_lengths",
        "surface_quadrature_weights", "reduced_memory_induction_matrix",
        "_periodic_induction_singular_values", "spectral_width",
        "smooth_minimum_distance", "smooth_minimum_tangent_radius",
        "enclosed_volume", "_calc_objectives", "calc_objectives",
        "_periodic_objectives", "_combine_winding_objectives",
        "_periodic_winding_objective_value",
        "_active_winding_objective_value", "_active_dof_bounds",
        "_solve_winding_surface", "_winding_linear_solve",
        "_solve_winding_surface_jvp",
    }
    tree = ast.parse(path.read_text(), filename=str(path))
    tree.body = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    ]
    namespace = {
        "functools": functools,
        "jax": jax,
        "jnp": jnp,
        "LBFGSB": LBFGSB,
        "root_jvp": root_jvp,
        "gcrot": gcrot,
        "MU0": 4 * np.pi * 1e-7,
        "nfp": 2,
        "WINDING_NPHI": 8,
        "WINDING_NTHETA": 8,
        "WINDING_MAXITER": 100,
        "WINDING_OPTIMALITY_TOL": 1e-6,
        "WINDING_LINEAR_RTOL": 1e-8,
        "WINDING_LINEAR_MAX_RESTARTS": 10,
        "COEFFICIENT_STEP_BOUND": 1.0,
        "DENOMINATOR_EPS": 1e-16,
        "SHARPNESS": 300.0,
        "SELF_NEIGHBOR_RADIUS": 2,
        "MINIMUM_DISTANCE": 0.05,
        "DISTANCE_WALL_SCALE": 0.01,
        "MINIMUM_SELF_RADIUS": 0.05,
        "SELF_RADIUS_WALL_SCALE": 0.01,
        "MINIMUM_NORMAL_LENGTH": 1e-6,
        "INVALID_PENALTY_SCALE": 1e12,
        "PCA_WEIGHT": 1.0,
        "SINGULAR_STRENGTH_WEIGHT": 0.0,
        "VOLUME_WEIGHT": 0.02,
        "SPECTRAL_WEIGHT": 0.02,
        "DISTANCE_WALL_WEIGHT": 10.0,
        "SELF_INTERSECTION_WEIGHT": 100.0,
    }
    exec(compile(tree, str(path), "exec"), namespace)

    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield namespace
    finally:
        jax.config.update("jax_enable_x64", previous)


def _torus_dofs(seed, *, major=3.0, minor=0.7):
    rng = np.random.default_rng(seed)
    rc = np.zeros((3, 5))
    zs = np.zeros((3, 5))
    rc[0, 2], rc[1, 2], zs[1, 2] = major, minor, minor
    dofs = np.r_[rc.ravel()[2:], zs.ravel()[3:]]
    return jnp.asarray(dofs + 2e-3 * rng.normal(size=dofs.shape)), jnp.asarray(rc)


@pytest.mark.parametrize("field_periods,tor_num,pol_num", [
    (1, 7, 5),
    (2, 8, 5),
    (3, 9, 4),
    (4, 8, 3),
    (2, 7, 5),  # non-divisible toroidal grid exercises the exact fallback
])
def test_periodic_spectrum_matches_full_matrix(
        winding_helpers, field_periods, tor_num, pol_num):
    helpers = winding_helpers
    winding, rc = _torus_dofs(20 + field_periods, minor=0.9)
    plasma, _ = _torus_dofs(40 + field_periods, minor=0.65)
    points = helpers["points_normals_normal_lengths"]
    wp, wn, wj, _ = points(
        winding, 2, 2, field_periods, tor_num, pol_num)
    pp, pn, pj, _ = points(plasma, 2, 2, field_periods, tor_num, pol_num)
    wp, wn = wp.reshape(-1, 3), wn.reshape(-1, 3)
    pp, pn = pp.reshape(-1, 3), pn.reshape(-1, 3)
    weights = helpers["surface_quadrature_weights"]
    ww = weights(wj, tor_num, pol_num)
    pw = weights(pj, tor_num, pol_num)

    matrix = helpers["reduced_memory_induction_matrix"](
        wp, pp, wn, pn, ww, pw)
    expected = jnp.linalg.svd(matrix, compute_uv=False)
    actual = helpers["_periodic_induction_singular_values"](
        wp, pp, wn, pn, ww, pw, field_periods, tor_num, pol_num)
    np.testing.assert_allclose(
        jnp.sort(actual), jnp.sort(expected), rtol=2e-11, atol=2e-15)


def test_periodic_entropy_value_gradient_and_hvp_match_full(
        winding_helpers, monkeypatch):
    helpers = winding_helpers
    monkeypatch.setitem(helpers, "nfp", 2)
    monkeypatch.setitem(helpers, "WINDING_NPHI", 8)
    monkeypatch.setitem(helpers, "WINDING_NTHETA", 6)
    winding, rc = _torus_dofs(12, minor=0.9)
    plasma, _ = _torus_dofs(31, minor=0.65)
    values = jnp.concatenate((winding, plasma))
    direction = jnp.asarray(np.random.default_rng(8).normal(size=values.shape))
    direction /= jnp.linalg.norm(direction)
    mode_weights = jnp.ones_like(winding)

    def entropy(value, periodic):
        winding_dofs, plasma_dofs = jnp.split(value, 2)
        if periodic:
            return helpers["_periodic_objectives"](
                winding_dofs, plasma_dofs, mode_weights, rc)[0]
        points = helpers["points_normals_normal_lengths"]
        pp, pn, pj, _ = points(plasma_dofs, 2, 2, 2, 8, 6)
        pw = helpers["surface_quadrature_weights"](pj, 8, 6)
        return helpers["calc_objectives"](
            winding_dofs, pp.reshape(-1, 3), pn.reshape(-1, 3),
            mode_weights, rc, pw)[0]

    def evaluate(periodic):
        def fun(value):
            return entropy(value, periodic)

        gradient = jax.grad(fun)
        return jax.jit(lambda value: (
            fun(value), gradient(value),
            jax.jvp(gradient, (value,), (direction,))[1]))(values)

    full, reduced = evaluate(False), evaluate(True)
    for expected, actual in zip(full, reduced):
        assert np.all(np.isfinite(actual))
        np.testing.assert_allclose(actual, expected, rtol=3e-8, atol=3e-10)


@pytest.mark.parametrize("matrix", [
    [[3.0, 1.0], [1.0, -2.0]],
    [[3.0, 2.0], [-1.0, 4.0]],
])
def test_winding_linear_solve_supports_forward_and_reverse(
        winding_helpers, matrix):
    matrix = jnp.asarray(matrix)
    rhs = jnp.array([0.7, -0.3])

    def solve(value):
        return winding_helpers["_winding_linear_solve"](
            lambda vector: matrix @ vector, value)

    expected = jnp.linalg.solve(matrix, rhs)
    np.testing.assert_allclose(jax.jit(solve)(rhs), expected, rtol=2e-12)

    direction = jnp.array([0.2, 0.8])
    tangent = jax.jvp(solve, (rhs,), (direction,))[1]
    cotangent = jax.grad(
        lambda value: jnp.dot(solve(value), direction))(rhs)
    np.testing.assert_allclose(
        tangent, jnp.linalg.solve(matrix, direction), rtol=2e-12)
    np.testing.assert_allclose(
        cotangent, jnp.linalg.solve(matrix.T, direction), rtol=2e-12)


def test_winding_root_response_is_transposable(winding_helpers, monkeypatch):
    helpers = winding_helpers
    matrix = jnp.array([[3.0, 0.4], [0.4, 2.0]])

    def objective(active, full, active_indices, parameter, *unused):
        del full, active_indices
        hessian = matrix + jnp.diag(parameter ** 2)
        return 0.5 * active @ hessian @ active - jnp.dot(parameter ** 2, active)

    monkeypatch.setitem(helpers, "_active_winding_objective_value", objective)
    indices = (0, 1)

    def solve(parameter):
        zeros = jnp.zeros(2)
        return helpers["_solve_winding_surface"](
            zeros, zeros, indices, parameter, zeros, zeros, zeros)

    def exact(parameter):
        return jnp.linalg.solve(
            matrix + jnp.diag(parameter ** 2), parameter ** 2)

    parameter = jnp.array([0.7, -0.3])
    direction = jnp.array([0.2, 0.8])
    value, tangent = jax.jvp(solve, (parameter,), (direction,))
    np.testing.assert_allclose(value, exact(parameter), atol=5e-7)
    np.testing.assert_allclose(
        tangent, jax.jvp(exact, (parameter,), (direction,))[1], atol=3e-7)

    gradient = jax.grad(lambda argument: jnp.sum(solve(argument)))
    expected_gradient = jax.grad(
        lambda argument: jnp.sum(exact(argument)))
    np.testing.assert_allclose(
        gradient(parameter), expected_gradient(parameter), atol=3e-7)
    np.testing.assert_allclose(
        jax.jvp(gradient, (parameter,), (direction,))[1],
        jax.jvp(expected_gradient, (parameter,), (direction,))[1],
        atol=5e-7)
