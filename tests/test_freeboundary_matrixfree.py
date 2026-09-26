"""Seed preconditioning must not reuse the seed operator or weaken acceptance."""

from types import SimpleNamespace as NS

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jl
import numpy as np
import pytest

from vmex.core import freeboundary_continuation as fc, freeboundary_implicit as fbi
from vmex.core import _freeboundary_dense as dense, _freeboundary_matrixfree as mf
from tests.test_freeboundary_linearization import fixture

pytestmark = pytest.mark.usefixtures("_module_jit_enabled")


@pytest.mark.parametrize("rhs_batch_size", [1, 3])
def test_current_nonlinear_root_arbitrary_rows_and_seed_lifetime(monkeypatch, rhs_batch_size):
    accepted, cfg, matrix, coupling = fixture(monkeypatch)
    residual = jax.jit(lambda z, p, f, *_: matrix @ z + 0.1 * z * z + coupling @ f - p)
    monkeypatch.setattr(fbi, "_projected_residual", lambda *_: residual)
    seed_root = fc.free_boundary_continuation_state_pullback(accepted, cfg, jnp.eye(2), return_linearization=True)
    seed_root.offload_factors()
    seed = seed_root.preconditioner(rhs_batch_size=rhs_batch_size)
    factors = seed._seed.factors[0].copy()
    seed_root.close()
    monkeypatch.setattr(dense, "_assemble_device", lambda *a, **k: pytest.fail("unexpected dense assembly"))
    state = jnp.array([0.2, -0.3])
    field = np.linalg.solve(coupling, -matrix @ state - 0.1 * state * state)
    other = NS(**(vars(accepted) | dict(state=state, parameters=field)))
    # Four unique rows also exercise the final, shorter batch.
    rhs = jnp.array([[1.0, 2.0], [3.0, 4.0], [-3.0, -4.0], [1.0, 2.0], [0.0, 0.0], [2.0, 5.0], [5.0, 1.0]])
    diagnostics = []
    root = fc.free_boundary_continuation_state_pullback(
        other, cfg, rhs, preconditioner=seed, return_linearization=True, diagnostics=diagnostics
    )
    current = np.asarray(matrix) + np.diag(0.2 * np.asarray(state))
    expected = -np.asarray(rhs) @ np.linalg.solve(current, coupling)
    np.testing.assert_allclose(root.field_jacobian, expected, rtol=1e-10, atol=1e-13)
    assert all(row["accepted"] for row in diagnostics)
    assert all(row["krylov_converged"] for row in diagnostics)
    assert diagnostics[4]["krylov_residual_norm"] == 0.0
    assert [row["iterations"] for row in diagnostics][2:5] == [0, 0, 0]
    direction = jnp.array([0.3, -0.4])
    notes = []
    for alpha in (1.0, 0.5, 0.0):
        np.testing.assert_allclose(
            root.tangent(other, cfg, alpha * direction, diagnostics=notes),
            -np.linalg.solve(current, coupling @ (alpha * direction)),
            rtol=1e-10,
            atol=1e-13,
        )
    assert notes[1]["scaled_reuse"] and notes[1]["iterations"] == 0
    np.testing.assert_array_equal(seed._seed.factors[0], factors)
    seed.close()
    # A live current-root predictor owns its factor reference independently.
    assert np.all(np.isfinite(root.tangent(other, cfg, direction)))
    with pytest.raises(ValueError, match="closed"):
        fc.free_boundary_continuation_state_pullback(other, cfg, rhs, preconditioner=seed)
    root.close()
    with pytest.raises(ValueError, match="closed"):
        root.tangent(other, cfg, direction)


def test_preconditioner_rejects_foreign_config_mask_and_uncertified_root(monkeypatch):
    accepted, cfg, _, _ = fixture(monkeypatch)
    root = fc.free_boundary_continuation_state_pullback(accepted, cfg, jnp.eye(2), return_linearization=True)
    seed = root.preconditioner()
    with pytest.raises(ValueError, match="different continuation"):
        fc.free_boundary_continuation_state_pullback(accepted, NS(**vars(cfg)), jnp.eye(2), preconditioner=seed)
    other = NS(**(vars(accepted) | dict(dof_mask=jnp.array([1.0, 0.0]))))
    with pytest.raises(ValueError, match="active space"):
        fc.free_boundary_continuation_state_pullback(other, cfg, jnp.eye(2), preconditioner=seed)
    other = NS(**(vars(accepted) | dict(state=jnp.ones(2))))
    with pytest.raises(ValueError, match="root residual"):
        fc.free_boundary_continuation_state_pullback(other, cfg, jnp.eye(2), preconditioner=seed)
    for kw in (
        dict(rtol=0.0),
        dict(rtol=np.nan),
        dict(restart=0),
        dict(max_restarts=1.5),
        dict(require_adjoint_convergence=1),
    ):
        with pytest.raises(ValueError):
            root.preconditioner(**kw)
    seed.close()
    root.close()


def test_exact_rhs_reuse_ignores_signed_zero_but_not_small_changes(monkeypatch):
    accepted, cfg, _, _ = fixture(monkeypatch)
    dense_root = fc.free_boundary_continuation_state_pullback(accepted, cfg, jnp.eye(2), return_linearization=True)
    seed = dense_root.preconditioner()
    rows = jnp.array([[1.0, 0.0], [-1.0, 0.0], [1.0, -0.0], [1.0, 1e-20]])
    reports = []
    result = fc.free_boundary_continuation_state_pullback(accepted, cfg, rows, preconditioner=seed, diagnostics=reports)
    assert [r["reused_rhs"] for r in reports] == [False, True, True, False]
    assert reports[1]["iterations"] == reports[2]["iterations"] == 0
    np.testing.assert_array_equal(np.asarray(result)[1], -np.asarray(result)[0])
    assert all(r["accepted"] for r in reports)
    dense_root.close()
    seed.close()


@pytest.mark.parametrize("rhs_batch_size", [1, 3])
def test_requested_convergence_rejects_under_solve_despite_broad_residual_pass(monkeypatch, rhs_batch_size):
    """A real one-iteration solve misses its request but passes the full 1e-6 gate."""
    accepted, cfg, _, coupling = fixture(monkeypatch)
    identity = jnp.eye(2)
    perturbation = jnp.array([[0.0, 1e-8], [0.0, 0.0]])
    residual = jax.jit(lambda z, p, f, *_: (identity + f[0] * perturbation) @ z + coupling @ f - p)
    monkeypatch.setattr(fbi, "_projected_residual", lambda *_: residual)
    cfg.solver.adjoint_residual_rtol = 1e-6
    rhs = jnp.array([[1.0, 1.0], [-1.0, -1.0], [0.0, 0.0]])
    original = fc.free_boundary_continuation_state_pullback(accepted, cfg, rhs, return_linearization=True)
    options = dict(rtol=1e-14, restart=1, max_restarts=1, rhs_batch_size=rhs_batch_size)
    legacy = original.preconditioner(**options)
    strict = original.preconditioner(**options, require_adjoint_convergence=True)
    matrix = np.asarray(identity + perturbation)
    point = NS(
        **(
            vars(accepted)
            | dict(
                parameters=np.array([1.0, 0.0]), state=jnp.asarray(np.linalg.solve(matrix, -np.asarray(coupling)[:, 0]))
            )
        )
    )
    reports = []
    fc.free_boundary_continuation_state_pullback(point, cfg, rhs, preconditioner=legacy, diagnostics=reports)
    assert all(r["accepted"] and r["full_residual_accepted"] for r in reports)
    assert [r["krylov_converged"] for r in reports] == [False, False, True]
    assert reports[0]["krylov_residual_norm"] == reports[1]["krylov_residual_norm"]
    assert reports[1]["reused_rhs"] and reports[1]["iterations"] == 0
    from vmex.core.errors import AdjointSolveError

    failure = []
    with pytest.raises(AdjointSolveError, match="requested Krylov tolerance was not met"):
        fc.free_boundary_continuation_state_pullback(point, cfg, rhs, preconditioner=strict, diagnostics=failure)
    assert failure[0]["full_residual_accepted"] and not failure[0]["accepted"]
    assert failure[0]["krylov_residual_norm"] > failure[0]["krylov_tolerance"]
    fresh = fc.free_boundary_continuation_state_pullback(point, cfg, rhs, return_linearization=True)
    refreshed = fresh.preconditioner(**options, require_adjoint_convergence=True)
    answer = fc.free_boundary_continuation_state_pullback(point, cfg, rhs, preconditioner=refreshed)
    np.testing.assert_allclose(answer, fresh.field_jacobian, rtol=1e-13, atol=1e-14)
    for item in (original, fresh, legacy, strict, refreshed):
        item.close()


def test_nonnormal_current_operator_rejects_short_solve_and_supports_two_recoveries(monkeypatch):
    """Exercise actual Krylov failure, a larger basis and fresh dense factors."""
    accepted, cfg, _, _ = fixture(monkeypatch)
    n = 96
    matrix = np.eye(n) - np.diag(np.ones(n - 1), -1)
    for offset in (1, 2, 3):
        matrix += np.diag(np.ones(n - offset), offset)
    identity, current = jnp.eye(n), jnp.asarray(matrix)
    coupling = jnp.asarray(np.random.default_rng(2).normal(size=(n, 2)))
    residual = jax.jit(lambda z, p, f, *_: (identity + f[0] * (current - identity)) @ z + coupling @ f - p)
    monkeypatch.setattr(fbi, "_projected_residual", lambda *_: residual)
    cfg.params = jnp.zeros(n)
    cfg.solver.adjoint_dense_max_dofs = 128
    cfg.solver.adjoint_dense_batch_size = 16
    cfg.solver.adjoint_residual_rtol = 1e-6
    accepted.state, accepted.dof_mask = jnp.zeros(n), jnp.ones(n)
    rhs = jnp.asarray(np.random.default_rng(0).normal(size=(1, n)))
    reference0 = fc.free_boundary_continuation_state_pullback(accepted, cfg, rhs, return_linearization=True)
    short = reference0.preconditioner(restart=30, max_restarts=1)
    larger = reference0.preconditioner(restart=100, max_restarts=3)
    reference0.close()
    state = jnp.asarray(np.linalg.solve(matrix, -np.asarray(coupling)[:, 0]))
    point = NS(**(vars(accepted) | dict(state=state, parameters=np.array([1.0, 0.0]))))
    reports = []
    from vmex.core.errors import AdjointSolveError

    with pytest.raises(AdjointSolveError):
        fc.free_boundary_continuation_state_pullback(point, cfg, rhs, preconditioner=short, diagnostics=reports)
    assert reports[0]["iterations"] == 30 and not reports[0]["accepted"]
    dense_reference = fc.free_boundary_continuation_state_pullback(point, cfg, rhs, return_linearization=True)
    field_matrix = np.column_stack(
        ((matrix - np.eye(n)) @ np.asarray(state) + np.asarray(coupling)[:, 0], np.asarray(coupling)[:, 1])
    )
    expected = -np.asarray(rhs) @ np.linalg.solve(matrix, field_matrix)
    np.testing.assert_allclose(dense_reference.field_jacobian, expected, rtol=1e-10, atol=1e-11)
    refreshed = dense_reference.preconditioner()
    for seed in (larger, refreshed):
        answer = fc.free_boundary_continuation_state_pullback(point, cfg, rhs, preconditioner=seed)
        np.testing.assert_allclose(answer, expected, rtol=1e-8, atol=1e-10)
    dense_reference.close()
    for seed in (short, larger, refreshed):
        seed.close()


def test_true_residual_rejects_bad_krylov_answer_in_both_directions(monkeypatch):
    accepted, cfg, _, _ = fixture(monkeypatch)
    dense_root = fc.free_boundary_continuation_state_pullback(accepted, cfg, jnp.eye(2), return_linearization=True)
    seed = dense_root.preconditioner()
    root = fc.free_boundary_continuation_state_pullback(
        accepted, cfg, jnp.eye(2), preconditioner=seed, return_linearization=True
    )
    monkeypatch.setattr(
        mf,
        "solve",
        lambda action, z, space, lu, rhs, **kw: (
            (jnp.zeros_like(rhs), 1, 0.0, True) if kw.get("return_info") else (jnp.zeros_like(rhs), 1)
        ),
    )
    reports = []
    with pytest.raises(Exception, match="matrixfree_seed_lu") as error:
        fc.free_boundary_continuation_state_pullback(
            accepted, cfg, jnp.eye(2), preconditioner=seed, diagnostics=reports
        )
    assert reports and not reports[0]["accepted"]
    assert "restart=2, max_cycles=10" in str(error.value)
    assert "max_restarts=300" not in str(error.value)
    assert reports[0]["requested_rtol"] == 1e-11
    with pytest.raises(Exception, match="matrixfree_seed_lu"):
        root.tangent(accepted, cfg, jnp.ones(2))
    root.close()
    seed.close()
    dense_root.close()


def test_nonfinite_full_residual_has_boolean_rejection_diagnostics(monkeypatch):
    accepted, cfg, _, _ = fixture(monkeypatch)
    reference = fc.free_boundary_continuation_state_pullback(accepted, cfg, jnp.eye(2), return_linearization=True)
    seed = reference.preconditioner()
    # A finite primal root can still have an undefined derivative. A bogus
    # finite Krylov answer must not turn the rejection into a string in JSON.
    monkeypatch.setattr(mf, "prepare", lambda *args, **kwargs: lambda value: value * jnp.nan)
    monkeypatch.setattr(mf, "solve", lambda *args, **kwargs: (jnp.ones(2), 1, 0.0, True))
    reports = []
    from vmex.core.errors import AdjointSolveError

    with pytest.raises(AdjointSolveError):
        fc.free_boundary_continuation_state_pullback(
            accepted, cfg, jnp.eye(2), preconditioner=seed, diagnostics=reports
        )
    assert reports[0]["accepted"] is False
    assert reports[0]["full_residual_accepted"] is False
    assert not np.isfinite(reports[0]["residual_norm"])
    seed.close()
    reference.close()


def test_stable_tape_changes_numbers_without_new_executables():
    matrix = jnp.array([[3.0, 1.0], [-0.5, 2.0]])

    @jax.jit
    def residual(z, p, f, b, r, c):
        return matrix @ z + 0.01 * z * z + f

    space = dense._Space(jnp.arange(2), jnp.arange(2), jnp.zeros(2), jnp.ones(2))
    factors = jl.lu_factor(matrix)
    counts = []
    for i in range(3):
        z = jnp.array([0.2, 0.4]) + i * 0.1
        action = mf.prepare(z, None, jnp.zeros(2), z, None, None, residual=residual)
        current = np.asarray(matrix) + np.diag(0.02 * np.asarray(z))
        for transpose in (False, True):
            result, _ = mf.solve(
                action, z, space, factors, jnp.ones(2), transpose=transpose, rtol=1e-11, restart=30, max_restarts=10
            )
            np.testing.assert_allclose(
                result, np.linalg.solve(current.T if transpose else current, np.ones(2)), rtol=1e-10
            )
        counts.append(mf.solve._cache_size())
    assert counts[0] == counts[1] == counts[2]


def test_redundant_paired_coordinates():
    space = dense._Space(jnp.array([0, 1]), jnp.array([0, 2]), jnp.array([0.0, -1.0]), jnp.array([1.0, 1 / np.sqrt(2)]))
    q = np.array([[1.0, 0.0], [0.0, 1 / np.sqrt(2)], [0.0, -1 / np.sqrt(2)]])
    active = np.array([[2.0, 0.5], [-0.8, 3.0]])
    action = jax.tree_util.Partial(jnp.matmul, jnp.asarray(q @ active @ q.T))
    factors = jl.lu_factor(jnp.asarray(active + np.eye(2) * 0.1))
    rhs = jnp.array([1.0, -2.0])
    for transpose in (False, True):
        solution, _ = mf.solve(
            action, jnp.zeros(3), space, factors, rhs, transpose=transpose, rtol=1e-11, restart=30, max_restarts=10
        )
        np.testing.assert_allclose(solution, np.linalg.solve(active.T if transpose else active, rhs), rtol=1e-10)


@pytest.mark.parametrize("size", [0, 5, True, 1.5])
def test_rhs_batch_size_is_bounded(monkeypatch, size):
    accepted, cfg, _, _ = fixture(monkeypatch)
    root = fc.free_boundary_continuation_state_pullback(accepted, cfg, jnp.eye(2), return_linearization=True)
    try:
        with pytest.raises(ValueError, match="rhs_batch_size"):
            root.preconditioner(rhs_batch_size=size)
    finally:
        root.close()


def test_batched_krylov_preserves_independent_convergence():
    matrix = jnp.diag(jnp.arange(1.0, 10.0))
    template = jnp.zeros(9)
    index = jnp.arange(9)
    space = dense._Space(index, index, jnp.zeros(9), jnp.ones(9))
    action = jax.tree_util.Partial(lambda a, x: a @ x, matrix)
    rhs = jnp.stack([jnp.eye(9)[0], jnp.ones(9), jnp.zeros(9)])
    x, iterations, norm, converged = mf._solve_many(
        action, template, space, jl.lu_factor(jnp.eye(9)), rhs, rtol=1e-11, restart=9, max_restarts=2
    )
    np.testing.assert_allclose(x, np.asarray(rhs) / np.arange(1.0, 10.0), rtol=1e-10, atol=1e-12)
    assert list(map(int, iterations))[0] == 1 and int(iterations[1]) > 1 and int(iterations[2]) == 0
    assert bool(jnp.all(converged)) and bool(jnp.all(norm < 1e-10))


@pytest.mark.parametrize("tangent_rtol, expected", [(None, 1e-9), (1e-11, 1e-11)])
def test_predictor_tolerance_is_independent_of_adjoint_request(monkeypatch, tangent_rtol, expected):
    accepted, cfg, _, _ = fixture(monkeypatch)
    dense_root = fc.free_boundary_continuation_state_pullback(accepted, cfg, jnp.eye(2), return_linearization=True)
    seed = dense_root.preconditioner(rtol=1e-9, tangent_rtol=tangent_rtol)
    calls = []
    original = mf.solve

    def capture(*args, **options):
        calls.append((options["transpose"], options["rtol"]))
        return original(*args, **options)

    monkeypatch.setattr(mf, "solve", capture)
    root = fc.free_boundary_continuation_state_pullback(
        accepted, cfg, jnp.eye(2), preconditioner=seed, return_linearization=True
    )
    root.tangent(accepted, cfg, jnp.ones(2))
    assert calls == [(True, 1e-9), (True, 1e-9), (False, expected)]
    root.close()
    seed.close()
    dense_root.close()


@pytest.mark.parametrize("value", [0.0, -1.0, 1.0, np.nan, np.inf])
def test_invalid_predictor_tolerance_is_rejected(monkeypatch, value):
    accepted, cfg, _, _ = fixture(monkeypatch)
    root = fc.free_boundary_continuation_state_pullback(accepted, cfg, jnp.eye(2), return_linearization=True)
    try:
        with pytest.raises(ValueError, match="tangent_rtol"):
            root.preconditioner(tangent_rtol=value)
    finally:
        root.close()


@pytest.mark.parametrize("rhs_batch_size", [1, 3])
def test_nonfinite_cotangents_are_rejected_before_krylov(monkeypatch, rhs_batch_size):
    accepted, cfg, _, _ = fixture(monkeypatch)
    root = fc.free_boundary_continuation_state_pullback(accepted, cfg, jnp.eye(2), return_linearization=True)
    seed = root.preconditioner(rhs_batch_size=rhs_batch_size)
    monkeypatch.setattr(mf, "solve", lambda *a, **k: pytest.fail("nonfinite RHS reached Krylov"))
    try:
        with pytest.raises(ValueError, match="nonfinite adjoint right-hand side"):
            fc.free_boundary_continuation_state_pullback(
                accepted, cfg, jnp.array([[1.0, jnp.nan]]), preconditioner=seed
            )
    finally:
        seed.close()
        root.close()


def test_closed_linearization_cannot_create_or_offload_seed(monkeypatch):
    accepted, cfg, _, _ = fixture(monkeypatch)
    root = fc.free_boundary_continuation_state_pullback(accepted, cfg, jnp.eye(2), return_linearization=True)
    dense_root = root._root
    root.close()
    for operation in (root.preconditioner, root.offload_factors):
        with pytest.raises(ValueError, match="closed"):
            operation()
    with pytest.raises(ValueError, match="live dense linearization"):
        mf.SeedLU.from_root(dense_root)


def test_seed_rejects_changed_solver_closed_storage_and_single_precision(monkeypatch):
    accepted, cfg, _, _ = fixture(monkeypatch)
    root = fc.free_boundary_continuation_state_pullback(accepted, cfg, jnp.eye(2), return_linearization=True)
    dense_root = root._root
    seed = root.preconditioner()
    with pytest.raises(ValueError, match="different solver or state layout"):
        seed._seed.validate(accepted.state, accepted.parameters, dense_root.space, NS(**vars(cfg.solver)))
    seed._seed.close()
    with pytest.raises(ValueError, match="closed"):
        seed._seed.validate(accepted.state, accepted.parameters, dense_root.space, cfg.solver)
    with pytest.raises(ValueError, match="Krylov rtol"):
        root.preconditioner(rtol=np.nan, tangent_rtol=1e-11)
    factors, pivots = dense_root.factors
    dense_root.factors = (factors.astype(jnp.float32), pivots)
    with pytest.raises(TypeError, match="float64"):
        root.preconditioner()
    seed.close()
    root.close()


@pytest.mark.parametrize("backend, failure", [("coupled_gcrot", "error"), ("forward_dense_jax", "best_effort")])
def test_seed_cannot_bypass_strict_backend_policy(monkeypatch, backend, failure):
    accepted, cfg, _, _ = fixture(monkeypatch)
    root = fc.free_boundary_continuation_state_pullback(accepted, cfg, jnp.eye(2), return_linearization=True)
    seed = root.preconditioner()
    cfg.solver.adjoint_solver, cfg.solver.adjoint_fail = backend, failure
    try:
        with pytest.raises(ValueError, match="seed LU requires"):
            fc.free_boundary_continuation_state_pullback(accepted, cfg, jnp.eye(2), preconditioner=seed)
    finally:
        seed.close()
        root.close()
