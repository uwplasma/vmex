"""Correctness checks for winding-surface spectral and implicit derivatives."""

import ast
import functools
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxopt import LBFGSB
from jaxopt.implicit_diff import root_jvp
from solvax import gcrot


REPO = Path(__file__).parents[1]


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


def test_periodic_pairwise_value_jacobian_and_hvp_match_full(
        winding_helpers, monkeypatch):
    """One source period preserves both geometric pairwise reductions."""
    helpers = winding_helpers
    monkeypatch.setitem(helpers, "nfp", 2)
    monkeypatch.setitem(helpers, "WINDING_NPHI", 8)
    monkeypatch.setitem(helpers, "WINDING_NTHETA", 6)
    winding, rc = _torus_dofs(82, minor=0.9)
    plasma, _ = _torus_dofs(93, minor=0.65)
    values = jnp.concatenate((winding, plasma))
    direction = jnp.asarray(np.random.default_rng(19).normal(size=values.shape))
    direction /= jnp.linalg.norm(direction)
    mode_weights = jnp.ones_like(winding)

    def pairwise(value, periodic):
        winding_dofs, plasma_dofs = jnp.split(value, 2)
        if periodic:
            objectives = helpers["_periodic_objectives"](
                winding_dofs, plasma_dofs, mode_weights, rc)
        else:
            points = helpers["points_normals_normal_lengths"]
            pp, pn, pj, _ = points(plasma_dofs, 2, 2, 2, 8, 6)
            pw = helpers["surface_quadrature_weights"](pj, 8, 6)
            objectives = helpers["calc_objectives"](
                winding_dofs, pp.reshape(-1, 3), pn.reshape(-1, 3),
                mode_weights, rc, pw)
        return jnp.stack((objectives[3], objectives[6]))

    def evaluate(periodic):
        def fun(value):
            return pairwise(value, periodic)

        jacobian = jax.jacrev(fun)
        return jax.jit(lambda value: (
            fun(value), jacobian(value),
            jax.jvp(jacobian, (value,), (direction,))[1]))(values)

    full, reduced = evaluate(False), evaluate(True)
    for expected, actual in zip(full, reduced):
        assert np.all(np.isfinite(actual))
        np.testing.assert_allclose(actual, expected, rtol=3e-8, atol=3e-10)


def test_normalization_scales_remain_in_total_derivative(winding_helpers):
    """The recomputed normalization reference must not be stop-gradient'd."""
    helpers = winding_helpers
    objectives = tuple(jnp.asarray(value) for value in (
        2.0, 3.0, 5.0, 0.2, 1.0, 7.0, 0.2,
    ))
    scales = jnp.asarray((4.0, 6.0, 10.0, 1.0, 1.0, 14.0, 1.0))

    actual = jax.grad(
        lambda reference: helpers["_combine_winding_objectives"](
            objectives, reference
        )
    )(scales)
    expected = jnp.zeros_like(scales).at[0].set(
        -helpers["PCA_WEIGHT"] * objectives[0] / scales[0] ** 2
    ).at[1].set(
        helpers["VOLUME_WEIGHT"] * objectives[1] / scales[1] ** 2
    ).at[2].set(
        -helpers["SPECTRAL_WEIGHT"] * objectives[2] / scales[2] ** 2
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-15)


def _git_blob(data):
    return subprocess.run(
        ["git", "hash-object", "--stdin"], cwd=REPO, input=data,
        capture_output=True, check=True,
    ).stdout.decode().strip()


def _snapshot_blob(spec):
    data = (REPO / spec["path"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == spec["sha256"]
    assert _git_blob(data) == spec["git_blob"]
    return spec["git_blob"]


def _recovered_patch_blob(spec, snapshots, worktree):
    snapshot = snapshots[spec["base_snapshot"]]
    target = worktree / snapshot["target_path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((REPO / snapshot["path"]).read_bytes())
    subprocess.run(["git", "init", "-q"], cwd=worktree, check=True)
    subprocess.run(
        ["git", "apply", "--whitespace=nowarn", REPO / spec["patch"]],
        cwd=worktree, check=True,
    )
    return _git_blob(target.read_bytes())


def test_winding_benchmark_provenance_is_reconstructible(tmp_path):
    manifest = json.loads(
        (REPO / "benchmarks" / "winding_surface_provenance.json").read_text()
    )
    snapshots = manifest["snapshots"]
    snapshot_blobs = {
        name: _snapshot_blob(spec) for name, spec in snapshots.items()
    }
    recoveries = manifest["recoveries"]
    for index, (name, spec) in enumerate(recoveries.items()):
        if spec["recovery"] == "snapshot":
            actual = snapshot_blobs[spec["snapshot"]]
        else:
            patch = REPO / spec["patch"]
            assert hashlib.sha256(patch.read_bytes()).hexdigest() == spec[
                "patch_sha256"
            ]
            actual = _recovered_patch_blob(
                spec, snapshots, tmp_path / f"recovery-{index}-{name}"
            )
        assert actual == spec["result_blob"]

    used_recoveries = set()
    for artifact, record in manifest["records"].items():
        raw = json.loads((REPO / artifact).read_text())
        provenance = raw["_provenance"]
        assert provenance["measurement_commit"] == record[
            "recorded_measurement_commit"
        ]
        generator = record["generator_recovery"]
        assert generator in recoveries
        used_recoveries.add(generator)
        source_commits = provenance["source_commits"]
        assert set(source_commits) == set(record["sources"])
        for label, source in record["sources"].items():
            assert source_commits[label] == source["recorded_commit"]
            assert source["recovery"] in recoveries
            used_recoveries.add(source["recovery"])
    assert used_recoveries == set(recoveries)


def test_winding_benchmark_svgs_are_valid_xml():
    paths = sorted((REPO / "benchmarks").glob("winding_surface_*.svg"))
    assert paths
    for path in paths:
        ET.parse(path)


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


def test_winding_linear_solve_checks_reported_true_residual(
        winding_helpers, monkeypatch):
    """A converged flag alone cannot bypass the residual certificate."""
    helpers = winding_helpers

    def inaccurate_gcrot(action, value, **unused):
        del action
        return SimpleNamespace(
            x=jnp.zeros_like(value),
            converged=jnp.asarray(True),
            residual_norm=2 * jnp.linalg.norm(value),
        )

    monkeypatch.setitem(helpers, "gcrot", inaccurate_gcrot)
    actual = helpers["_winding_linear_solve"](
        lambda vector: vector, jnp.array([1.0, -2.0]))
    assert np.all(np.isnan(actual))


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
