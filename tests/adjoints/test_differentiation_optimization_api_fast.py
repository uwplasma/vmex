from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax.optimization import FixedBoundaryExactOptimizer


def _quadratic_stub_optimizer(*, n_params: int = 1) -> FixedBoundaryExactOptimizer:
    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._solver_device_name = None
    opt._inside_solver_device_context = False
    opt._scan_exact_path = "tape"
    opt._exact_cache = {}
    opt._exact_state_cache = {}
    opt._exact_state_key_by_id = {}
    opt._exact_residual_cache = {}
    opt._exact_jacobian_cache = {}
    opt._trial_residual_cache = OrderedDict()
    opt._profile = {}
    opt._last_jacobian_key = [None]
    opt._last_jacobian_residual = None
    opt._last_jacobian_source = "test"
    opt._n_non_qs = 1
    opt._n_qs = None
    opt._has_residual_block_metadata = True
    opt._qs_total_from_state_fn = None
    opt._aspect_target = 5.0
    opt._aspect_weight = 1.0
    opt._inner_max_iter = 3
    opt._inner_ftol = 1.0e-9
    opt._trial_max_iter = 2
    opt._trial_ftol = 1.0e-6
    opt._static = SimpleNamespace(cfg=SimpleNamespace(lasym=False))
    opt._base_params_vector = lambda: np.zeros(n_params)

    def residual_for_params(params):
        params = np.asarray(params, dtype=float).reshape(-1)
        return np.concatenate(([params[0] + 0.1], params[1:] if params.size > 1 else [0.0]))

    def residual_for_state(state):
        return residual_for_params(np.asarray(state.params, dtype=float))

    def solve_exact(params, return_payload=False):
        state = SimpleNamespace(params=np.asarray(params, dtype=float).copy())
        opt._remember_exact_state(opt._exact_cache_key(params), state)
        if return_payload:
            return state, {"tape": object(), "axis_override": {}}
        return state

    opt._residuals_fn = residual_for_state
    opt._residuals_eval_fn = residual_for_state
    opt._solve_exact_with_tape = solve_exact
    opt._solve_scan_exact_state = lambda params: solve_exact(params, return_payload=False)
    opt.residual_fun = lambda params: opt._remember_exact_residual(
        opt._exact_cache_key(params), residual_for_params(params)
    ) or residual_for_params(params)
    opt.forward_residual_fun = lambda params: residual_for_params(params) + 10.0
    opt.jacobian_fun = lambda params: np.eye(len(np.asarray(params).reshape(-1)))
    return opt


def test_run_selects_best_exact_accepted_point_for_final_outputs(monkeypatch) -> None:
    import scipy.optimize

    opt = _quadratic_stub_optimizer()
    callbacks = {}

    def fake_least_squares(fun, x0, jac, **kwargs):
        callbacks["initial_residual"] = fun(np.asarray(x0, dtype=float))
        callbacks["initial_jacobian"] = jac(np.asarray(x0, dtype=float))
        return SimpleNamespace(
            x=np.asarray([1.0]),
            cost=0.5 * 1.1**2,
            nfev=2,
            njev=1,
            success=True,
            status=1,
            message="fake scipy accepted a worse final trial",
        )

    monkeypatch.setattr(scipy.optimize, "least_squares", fake_least_squares)

    result = opt.run(np.asarray([0.0]), method="scipy", max_nfev=2, verbose=0)

    np.testing.assert_allclose(result["x"], [0.0])
    assert result["_state_final"].params[0] == pytest.approx(0.0)
    assert result["cost"] == pytest.approx(0.5 * 0.1**2)
    assert result["_history_dump"]["objective_final"] == pytest.approx(0.1**2)
    assert result["_history_dump"]["selected_best_exact_point"] is True
    np.testing.assert_allclose(callbacks["initial_jacobian"], [[1.0]])


def test_run_reuses_retained_best_exact_state_when_final_replay_fails(monkeypatch) -> None:
    import scipy.optimize

    opt = _quadratic_stub_optimizer()
    replay_attempts = []

    def fake_least_squares(fun, x0, jac, **kwargs):
        fun(np.asarray(x0, dtype=float))
        jac(np.asarray(x0, dtype=float))
        opt._exact_state_cache.clear()

        def fail_final_replay(params, return_payload=False):
            replay_attempts.append(np.asarray(params, dtype=float).copy())
            raise RuntimeError("synthetic final replay failure")

        opt._solve_exact_with_tape = fail_final_replay
        return SimpleNamespace(
            x=np.asarray([1.0]),
            cost=0.5 * 1.1**2,
            nfev=2,
            njev=1,
            success=True,
            status=1,
            message="fake scipy returned worse final point",
        )

    monkeypatch.setattr(scipy.optimize, "least_squares", fake_least_squares)

    result = opt.run(np.asarray([0.0]), method="scipy", max_nfev=2, verbose=0)

    np.testing.assert_allclose(result["x"], [0.0])
    assert result["_state_final"].params[0] == pytest.approx(0.0)
    assert result["_history_dump"]["selected_best_exact_point"] is True
    assert len(replay_attempts) == 1
    np.testing.assert_allclose(replay_attempts[0], [1.0])


def test_remember_best_exact_point_ignores_mismatched_supplied_state() -> None:
    opt = _quadratic_stub_optimizer()
    params = np.asarray([0.0])
    matching_state = SimpleNamespace(params=params.copy())
    stale_state = SimpleNamespace(params=np.asarray([99.0]))
    params_key = opt._exact_cache_key(params)
    stale_key = opt._exact_cache_key(np.asarray([99.0]))
    opt._exact_state_cache = {params_key: matching_state}
    opt._exact_state_key_by_id = {id(matching_state): params_key, id(stale_state): stale_key}

    opt._remember_best_exact_point(params, np.asarray([0.1, 0.0]), state=stale_state)

    assert opt._best_exact_state is matching_state


def test_matrix_free_branch_routes_scaled_linear_operator_products(monkeypatch) -> None:
    import scipy.optimize
    from scipy.sparse.linalg import LinearOperator

    opt = _quadratic_stub_optimizer(n_params=2)
    matrix = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    seen = {}

    def residual_linear_operator(params):
        seen["operator_params"] = np.asarray(params, dtype=float).copy()
        return LinearOperator(
            shape=matrix.shape,
            matvec=lambda v: matrix @ np.asarray(v, dtype=float),
            matmat=lambda v: matrix @ np.asarray(v, dtype=float),
            rmatvec=lambda w: matrix.T @ np.asarray(w, dtype=float),
            dtype=np.dtype(float),
        )

    opt.residual_linear_operator = residual_linear_operator

    def fake_least_squares(fun, x0, jac, **kwargs):
        seen["residual_at_y0"] = fun(np.asarray(x0, dtype=float))
        op = jac(np.asarray(x0, dtype=float))
        seen["matvec"] = op.matvec(np.asarray([2.0, 4.0]))
        seen["matmat"] = op.matmat(np.asarray([[2.0, 1.0], [4.0, 3.0]]))
        seen["rmatvec"] = op.rmatvec(np.asarray([5.0, 6.0]))
        seen["tr_solver"] = kwargs["tr_solver"]
        return SimpleNamespace(
            x=np.asarray(x0, dtype=float),
            cost=0.5 * float(np.dot(seen["residual_at_y0"], seen["residual_at_y0"])),
            nfev=1,
            njev=1,
            success=True,
            status=1,
            message="matrix-free ok",
        )

    monkeypatch.setattr(scipy.optimize, "least_squares", fake_least_squares)

    result = opt.run(
        np.asarray([0.0, 0.0]),
        method="scipy_matrix_free",
        x_scale=np.asarray([10.0, 0.5]),
        max_nfev=1,
        verbose=0,
    )

    np.testing.assert_allclose(seen["operator_params"], [0.0, 0.0])
    np.testing.assert_allclose(seen["matvec"], matrix @ np.asarray([20.0, 2.0]))
    np.testing.assert_allclose(seen["matmat"], matrix @ np.asarray([[20.0, 10.0], [2.0, 1.5]]))
    np.testing.assert_allclose(seen["rmatvec"], (matrix.T @ np.asarray([5.0, 6.0])) * np.asarray([10.0, 0.5]))
    assert seen["tr_solver"] == "lsmr"
    assert result["_history_dump"]["method"] == "scipy_matrix_free"


def test_scalar_trust_branch_uses_objective_gradient_and_records_best_exact() -> None:
    opt = _quadratic_stub_optimizer()
    calls = []

    def scalar_residual(params):
        params = np.asarray(params, dtype=float)
        residual = np.asarray([params[0] - 1.0, 0.0])
        opt._remember_exact_residual(opt._exact_cache_key(params), residual)
        return residual

    opt.residual_fun = scalar_residual
    opt._residuals_fn = lambda state: scalar_residual(state.params)
    opt._residuals_eval_fn = opt._residuals_fn

    def objective_and_gradient(params):
        params = np.asarray(params, dtype=float)
        calls.append(params.copy())
        residual = params[0] - 1.0
        opt._remember_exact_residual(opt._exact_cache_key(params), np.asarray([residual, 0.0]))
        opt._remember_exact_state(opt._exact_cache_key(params), SimpleNamespace(params=params.copy()))
        return 0.5 * residual**2, np.asarray([residual])

    opt.objective_and_gradient_fun = objective_and_gradient

    result = opt.run(
        np.asarray([0.0]),
        method="scalar_trust",
        scalar_step_bound=0.25,
        max_nfev=3,
        ftol=0.0,
        gtol=1.0e-12,
        verbose=0,
    )

    assert len(calls) == 3
    np.testing.assert_allclose(result["x"], [0.5])
    assert result["cost"] == pytest.approx(0.5 * 0.5**2)
    assert result["nit"] == 2
    assert result["_history_dump"]["method"] == "scalar_trust"
    assert result["_history_dump"]["objective_final"] < result["_history_dump"]["objective_initial"]


def test_least_squares_problem_routes_plain_state_and_qi_objectives() -> None:
    import vmec_jax.optimization_workflow as workflow

    options = workflow.QuasiIsodynamicOptions(surfaces=[0.5])

    class StateObjective:
        name = "state_obj"

        def J(self, _ctx, _state):
            return 5.0

        def to_objective_term(self, *, target, residual_weight):
            return workflow.ObjectiveTerm(
                self.name,
                self.J,
                target=target,
                weight=residual_weight,
                metadata={"state_target": float(target)},
            )

    class QIObjective:
        requires_qi_field = True

        def __init__(self, qi_options):
            self.qi_options = qi_options

        def J(self, _ctx, _state):
            raise RuntimeError("QI objective should route through to_qi_term")

        def to_qi_term(self, residual_weight):
            return workflow.QIObjectiveTerm(
                "qi_obj",
                lambda _ctx, _state, _field: (residual_weight * np.asarray([1.0, 2.0]), 5.0),
                qi_options=self.qi_options,
            )

    def plain(_ctx, _state):
        return np.asarray([3.0, 5.0])

    problem = workflow.LeastSquaresProblem.from_tuples(
        [
            (StateObjective().J, 1.0, 4.0),
            (QIObjective(options).J, 0.0, 9.0),
            (plain, 2.0, 16.0),
        ]
    )

    assert problem.is_qi is True
    assert problem.qi_options is options
    assert problem.metadata == {"state_target": 1.0}
    assert [term.name for term in problem.objective_terms] == ["state_obj", "plain"]
    np.testing.assert_allclose(problem.objective_terms[0].residual(None, None), [8.0])
    np.testing.assert_allclose(problem.objective_terms[1].residual(None, None), [4.0, 12.0])
    np.testing.assert_allclose(problem.qi_objective_terms[0].residual_and_total(None, None, {})[0], [3.0, 6.0])

    with pytest.raises(ValueError, match="target=0"):
        workflow.LeastSquaresProblem.from_tuples([(QIObjective(options).J, 1.0, 1.0)])

    with pytest.raises(ValueError, match="share one QuasiIsodynamicOptions"):
        workflow.LeastSquaresProblem.from_tuples(
            [
                (QIObjective(options).J, 0.0, 1.0),
                (QIObjective(workflow.QuasiIsodynamicOptions(surfaces=[0.5])).J, 0.0, 1.0),
            ]
        )
