from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

import vmec_jax.optimization as opt_module
from vmec_jax.boundary import BoundaryCoeffs
from vmec_jax.modes import ModeTable
from vmec_jax.optimization import (
    FixedBoundaryExactOptimizer,
    _indexed_boundary_maps_from_boundary,
    boundary_param_specs,
    gauss_newton_least_squares,
    make_qh_residuals_fn,
    make_qs_residuals_fn,
)
from vmec_jax.state import StateLayout, VMECState, pack_state


def _state_from_coeffs(r=1.0, rs=0.0, z=0.0, zs=0.0, l=0.0, ls=0.0) -> VMECState:
    import jax.numpy as jnp

    layout = StateLayout(ns=1, K=1, lasym=False)
    return VMECState(
        layout=layout,
        Rcos=jnp.asarray([[r]], dtype=jnp.float64),
        Rsin=jnp.asarray([[rs]], dtype=jnp.float64),
        Zcos=jnp.asarray([[z]], dtype=jnp.float64),
        Zsin=jnp.asarray([[zs]], dtype=jnp.float64),
        Lcos=jnp.asarray([[l]], dtype=jnp.float64),
        Lsin=jnp.asarray([[ls]], dtype=jnp.float64),
    )


def test_boundary_specs_and_indexed_maps_skip_negative_m_modes() -> None:
    modes = ModeTable(m=np.asarray([-1, 0, 1], dtype=int), n=np.asarray([0, 0, -1], dtype=int))
    boundary = BoundaryCoeffs(
        R_cos=np.asarray([99.0, 1.0, 2.0]),
        R_sin=np.asarray([88.0, 0.0, 0.2]),
        Z_cos=np.asarray([77.0, 0.0, 0.3]),
        Z_sin=np.asarray([66.0, 0.0, 0.4]),
    )

    specs = boundary_param_specs(
        boundary,
        modes,
        include=("rc", "rs", "zc", "zs"),
        include_axis=True,
        fix=(),
    )
    maps = _indexed_boundary_maps_from_boundary(boundary, modes)

    assert all(spec.index != 0 for spec in specs)
    assert (0, -1) not in maps["RBC"]
    assert maps["RBC"][(0, 0)] == pytest.approx(1.0)
    assert maps["ZBS"][(-1, 1)] == pytest.approx(0.4)


def test_least_squares_problem_rejects_malformed_simsopt_tuples() -> None:
    from vmec_jax.optimization_workflow import LeastSquaresProblem

    def objective(_ctx, _state):
        return np.asarray([1.0])

    problem = LeastSquaresProblem.from_tuples([(objective, 0.0, 4.0)])
    np.testing.assert_allclose(problem.objective_terms[0].residual(None, None), [2.0])

    with pytest.raises(ValueError, match="objective tuples must be"):
        LeastSquaresProblem.from_tuples([(objective, 0.0)])
    with pytest.raises(TypeError, match="first entry must be callable"):
        LeastSquaresProblem.from_tuples([("not-callable", 0.0, 1.0)])


def test_gauss_newton_verbose_termination_and_damping_edges(monkeypatch, capsys) -> None:
    gtol_result = gauss_newton_least_squares(
        lambda _x: np.asarray([0.0]),
        lambda _x: np.asarray([[1.0]]),
        np.asarray([0.0]),
        max_nfev=1,
        gtol=1.0,
        verbose=1,
    )
    assert gtol_result["success"] is True

    xtol_result = gauss_newton_least_squares(
        lambda _x: np.asarray([0.1]),
        lambda _x: np.asarray([[1.0]]),
        np.asarray([0.0]),
        max_nfev=1,
        gtol=0.0,
        xtol=1.0,
        verbose=1,
    )
    assert xtol_result["message"] == "`xtol` termination condition is satisfied."

    accepted_result = gauss_newton_least_squares(
        lambda x: np.asarray([float(x[0]) - 1.0]),
        lambda _x: np.asarray([[1.0]]),
        np.asarray([0.0]),
        max_nfev=4,
        gtol=0.0,
        xtol=0.0,
        verbose=1,
    )
    assert accepted_result["cost"] == pytest.approx(0.0)

    def raise_solve(*_args, **_kwargs):
        raise np.linalg.LinAlgError("synthetic damped solve failure")

    monkeypatch.setattr(opt_module.np.linalg, "solve", raise_solve)
    failed_result = gauss_newton_least_squares(
        lambda _x: np.asarray([1.0]),
        lambda _x: np.asarray([[1.0]]),
        np.asarray([0.0]),
        max_nfev=20,
        gtol=0.0,
        xtol=0.0,
        damping_factors=(0.0, -1.0, 1.0e-6),
        verbose=1,
    )
    assert failed_result["message"] == "line search failed to reduce the objective"

    out = capsys.readouterr().out
    assert "Iteration" in out
    assert "Optimality" in out


def test_gauss_newton_skips_nonfinite_damped_step(monkeypatch) -> None:
    monkeypatch.setattr(opt_module.np.linalg, "solve", lambda *_args, **_kwargs: np.asarray([np.inf]))

    result = gauss_newton_least_squares(
        lambda _x: np.asarray([1.0]),
        lambda _x: np.asarray([[1.0]]),
        np.asarray([0.0]),
        max_nfev=20,
        gtol=0.0,
        xtol=0.0,
        damping_factors=(1.0,),
        verbose=0,
    )

    assert result["success"] is False
    assert result["message"] == "line search failed to reduce the objective"


def test_qh_qs_residual_factories_fallback_sign_and_min_abs_iota_cotangent(monkeypatch) -> None:
    import jax.numpy as jnp
    import vmec_jax.boundary as boundary_module
    import vmec_jax.modes as modes_module
    import vmec_jax.quasisymmetry as qs_module
    import vmec_jax.wout as wout_module

    static = SimpleNamespace(
        s=np.asarray([0.0, 1.0]),
        modes=ModeTable(m=np.asarray([0]), n=np.asarray([0])),
        cfg=SimpleNamespace(mpol=1, ntor=0, ntheta=3, nzeta=3, nfp=1),
    )
    state = _state_from_coeffs(r=2.0, rs=0.3, z=0.5)

    monkeypatch.setattr(
        boundary_module,
        "boundary_from_indata",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("force sign fallback")),
    )
    monkeypatch.setattr(opt_module, "flux_profiles_from_indata", lambda *_args, **_kwargs: "flux")
    monkeypatch.setattr(opt_module, "_pressure_profile_for_static", lambda *_args, **_kwargs: jnp.asarray([0.0, 1.0]))
    monkeypatch.setattr(
        modes_module,
        "nyquist_mode_table_from_grid",
        lambda **_kwargs: SimpleNamespace(m=np.asarray([0]), n=np.asarray([0])),
    )
    monkeypatch.setattr(qs_module, "_quasisymmetry_angle_cache", lambda **_kwargs: "angle-cache")

    def fake_qs_ratio(**kwargs):
        assert kwargs["signgs"] == 1
        return {
            "residuals1d": jnp.asarray([kwargs["state"].Zcos[0, 0]], dtype=jnp.float64),
            "total": kwargs["state"].Zcos[0, 0] ** 2,
        }

    monkeypatch.setattr(qs_module, "quasisymmetry_ratio_residual_from_state", fake_qs_ratio)
    monkeypatch.setattr(
        wout_module,
        "equilibrium_aspect_ratio_from_state",
        lambda *, state, static: state.Rcos[0, 0] + 1.0,
    )
    monkeypatch.setattr(
        wout_module,
        "equilibrium_iota_profiles_from_state",
        lambda **kwargs: (
            None,
            jnp.asarray([0.0, kwargs["state"].Rsin[0, 0] + 0.2], dtype=jnp.float64),
            None,
        ),
    )

    qh = make_qh_residuals_fn(static, object(), target_aspect=2.5, aspect_weight=2.0, qs_weight=3.0)
    np.testing.assert_allclose(np.asarray(qh(state)), [1.0, 1.5])
    assert qh._objective_family == "qs"
    assert qh._helicity_m == 1
    assert qh._helicity_n == -1

    qs = make_qs_residuals_fn(
        static,
        object(),
        min_abs_iota=0.7,
        iota_floor_softness=0.25,
        iota_weight=4.0,
        qs_weight=2.0,
        surfaces=[0.5],
    )
    residual = np.asarray(qs(state), dtype=float)
    cotangent = qs._state_cotangent_from_packed(pack_state(state), state.layout, jnp.asarray([1.0, 1.0]))

    assert qs._objective_family == "qs"
    assert qs._helicity_m == 1
    assert qs._helicity_n == 0
    assert residual.shape == (2,)
    assert residual[0] > 0.0
    assert np.all(np.isfinite(np.asarray(cotangent)))


def test_qs_residual_factory_objective_helper_matches_weighted_residual_cost(monkeypatch) -> None:
    import jax.numpy as jnp
    import vmec_jax.modes as modes_module
    import vmec_jax.quasisymmetry as qs_module
    import vmec_jax.wout as wout_module

    static = SimpleNamespace(
        s=np.asarray([0.0, 1.0]),
        modes=ModeTable(m=np.asarray([0]), n=np.asarray([0])),
        cfg=SimpleNamespace(mpol=1, ntor=0, ntheta=3, nzeta=3, nfp=1),
    )
    state = _state_from_coeffs(r=2.0, rs=0.4, z=0.5)

    monkeypatch.setattr(opt_module, "flux_profiles_from_indata", lambda *_args, **_kwargs: "flux")
    monkeypatch.setattr(opt_module, "_pressure_profile_for_static", lambda *_args, **_kwargs: jnp.asarray([0.0, 1.0]))
    monkeypatch.setattr(
        modes_module,
        "nyquist_mode_table_from_grid",
        lambda **_kwargs: SimpleNamespace(m=np.asarray([0]), n=np.asarray([0])),
    )
    monkeypatch.setattr(qs_module, "_quasisymmetry_angle_cache", lambda **_kwargs: "angle-cache")

    def fake_qs_ratio(**kwargs):
        state_arg = kwargs["state"]
        residual = state_arg.Zcos[0, 0] + 2.0 * state_arg.Rcos[0, 0]
        return {
            "residuals1d": jnp.asarray([residual], dtype=jnp.float64),
            "total": residual * residual,
        }

    monkeypatch.setattr(qs_module, "quasisymmetry_ratio_residual_from_state", fake_qs_ratio)
    monkeypatch.setattr(
        wout_module,
        "equilibrium_aspect_ratio_from_state",
        lambda *, state, static: state.Rcos[0, 0] + 1.0,
    )
    monkeypatch.setattr(
        wout_module,
        "equilibrium_iota_profiles_from_state",
        lambda **kwargs: (
            None,
            jnp.asarray([0.0, kwargs["state"].Rsin[0, 0] + 0.2], dtype=jnp.float64),
            None,
        ),
    )

    residuals_fn = make_qs_residuals_fn(
        static,
        object(),
        signgs=1,
        target_aspect=2.5,
        target_iota=0.7,
        aspect_weight=2.0,
        iota_weight=3.0,
        qs_weight=4.0,
        surfaces=[0.5],
    )

    residual = np.asarray(residuals_fn(state), dtype=float)
    value, cotangent = residuals_fn._state_objective_value_and_cotangent_from_packed(
        pack_state(state),
        state.layout,
    )

    np.testing.assert_allclose(float(np.asarray(value)), 0.5 * float(np.dot(residual, residual)))
    assert residuals_fn._n_non_qs == 2
    assert np.all(np.isfinite(np.asarray(cotangent)))
    assert np.asarray(cotangent).shape == np.asarray(pack_state(state)).shape


def test_device_selection_context_move_and_recursive_dispatch_branches(monkeypatch) -> None:
    import vmec_jax._compat as compat

    optimizer = object.__new__(FixedBoundaryExactOptimizer)
    optimizer._solver_device_name = None
    assert optimizer._select_exact_path() == "tape"

    monkeypatch.setattr(
        compat,
        "jax",
        SimpleNamespace(default_backend=lambda: (_ for _ in ()).throw(RuntimeError("backend unavailable"))),
    )
    assert optimizer._select_exact_path() == "tape"

    optimizer._solver_device_name = "cpu"
    monkeypatch.setattr(compat, "jax", None)
    assert isinstance(optimizer._solver_device_context(), nullcontext)
    payload = {"x": np.asarray([1.0])}
    assert optimizer._move_to_solver_device(payload) is payload

    monkeypatch.setattr(
        compat,
        "jax",
        SimpleNamespace(devices=lambda _name: (_ for _ in ()).throw(RuntimeError("device probe failed"))),
    )
    assert isinstance(optimizer._solver_device_context(), nullcontext)
    assert optimizer._move_to_solver_device(payload) is payload

    class FakeArray:
        pass

    monkeypatch.setattr(
        compat,
        "jax",
        SimpleNamespace(
            Array=FakeArray,
            devices=lambda _name: ["cpu:0"],
            device_put=lambda obj, device: ("put", np.asarray(obj).tolist(), device),
        ),
    )
    moved_tuple = optimizer._move_to_solver_device((np.asarray([1.0, 2.0]), "kept"))
    assert moved_tuple == (("put", [1.0, 2.0], "cpu:0"), "kept")

    dispatched = object.__new__(FixedBoundaryExactOptimizer)
    dispatched._solver_device_name = "gpu"
    dispatched._inside_solver_device_context = False
    dispatched._run_in_solver_device_context = lambda fn, *args, **kwargs: (fn.__name__, args, kwargs)

    assert dispatched._solve_forward(np.asarray([0.0]), trial=True)[0] == "_solve_forward"
    assert dispatched._solve_exact_with_tape(np.asarray([0.0]), return_payload=True)[0] == "_solve_exact_with_tape"
    assert dispatched.residual_fun(np.asarray([0.0]))[0] == "residual_fun"
    assert dispatched.forward_residual_fun(np.asarray([0.0]))[0] == "forward_residual_fun"
    assert dispatched.jacobian_fun(np.asarray([0.0]))[0] == "jacobian_fun"
    assert dispatched.state_tangent_columns_fun(np.asarray([0.0]))[0] == "state_tangent_columns_fun"
    assert dispatched.b_cartesian_tangent_columns_fun(np.asarray([0.0]))[0] == "b_cartesian_tangent_columns_fun"
    assert dispatched.objective_and_gradient_fun(np.asarray([0.0]))[0] == "objective_and_gradient_fun"
    assert dispatched.residual_linear_operator(np.asarray([0.0]))[0] == "residual_linear_operator"


def test_optimizer_profile_cache_and_history_small_branches(monkeypatch) -> None:
    import vmec_jax._compat as compat
    import vmec_jax.init_guess as init_guess_module

    optimizer = object.__new__(FixedBoundaryExactOptimizer)
    optimizer._profile = {}

    assert optimizer._profile_solver_timing({"timing": "not-a-dict"}, profile_prefix="bad", phase_wall_s=1.0, unattributed_name=None) == 0.0
    solver_total = optimizer._profile_solver_timing(
        {"timing": {"solve_total_s": object(), "compute_forces_s": 0.2}},
        profile_prefix="solver",
        phase_wall_s=0.5,
        unattributed_name="solver_unattributed",
    )
    assert solver_total == pytest.approx(0.2)

    optimizer._profile_exact_tape_solver_timing(
        SimpleNamespace(diagnostics={"timing": {"tape_solve_call_s": object()}}),
        tape_build_wall_s=0.1,
    )

    optimizer._profile = {}
    monkeypatch.setenv("VMEC_JAX_OPT_SYNC_REPLAY_TIMING", "1")
    monkeypatch.setattr(compat, "jax", SimpleNamespace(block_until_ready=lambda value: ("ready", value)))
    assert optimizer._profile_async_phase("phase", opt_module.time.perf_counter(), "value") == ("ready", "value")
    monkeypatch.setattr(
        compat,
        "jax",
        SimpleNamespace(block_until_ready=lambda _value: (_ for _ in ()).throw(RuntimeError("sync failed"))),
    )
    assert optimizer._profile_async_phase("phase_fail", opt_module.time.perf_counter(), "value") == "value"

    monkeypatch.setenv("VMEC_JAX_OPT_JIT_RESIDUALS", "0")
    residuals_fn = lambda state: state
    assert optimizer._make_residuals_eval_fn(residuals_fn) is residuals_fn

    optimizer._callback_point_ids = None
    assert optimizer._callback_point_id(b"same") == 0
    assert optimizer._callback_point_id(b"same") == 0

    monkeypatch.setenv("VMEC_JAX_OPT_JIT_INITIAL_STATE", "0")
    assert optimizer._initial_state_from_params_jit(np.asarray([0.0])) is None

    assert optimizer._exact_history_accepts(np.inf) is False

    trial_owner = object.__new__(FixedBoundaryExactOptimizer)
    trial_owner._exact_cache_key = lambda params: np.asarray(params, dtype=float).reshape(-1).tobytes()
    trial_owner._trial_residual_cache_max = 2
    trial_owner._remember_trial_residual(np.asarray([1.0]), np.asarray([2.0]))
    assert isinstance(trial_owner._trial_residual_cache, OrderedDict)

    tangent_owner = object.__new__(FixedBoundaryExactOptimizer)
    tangent_owner._boundary_from_params = lambda params: ("boundary", tuple(np.asarray(params, dtype=float)))
    tangent_owner._boundary_input = None
    tangent_owner._static = SimpleNamespace(
        cfg=SimpleNamespace(lasym=False, ns=3),
        modes=SimpleNamespace(K=2),
    )
    monkeypatch.setattr(init_guess_module, "_vmec_lflip_from_boundary", lambda *_args, **_kwargs: None)
    assert tangent_owner._initial_tangent_cache_key(np.asarray([0.0, 1.0])) == (2, False, False, False, 3, 2)

    qs_owner = object.__new__(FixedBoundaryExactOptimizer)
    qs_owner._n_qs = 2
    assert qs_owner._qs_from_res(np.asarray([9.0, 3.0, 4.0])) == pytest.approx(25.0)
    qs_owner._aspect_target = 7.0
    qs_owner._aspect_weight = 0.0
    assert qs_owner._can_build_aspect_from_residuals() is False
    qs_owner._qs_total_from_state_fn = None
    qs_owner._evaluate_residuals_from_state = lambda _state: np.asarray([1.0, 2.0, 2.0])
    qs_owner._n_qs = None
    qs_owner._n_non_qs = 1
    assert qs_owner._qs_total_from_state("state", None) == pytest.approx(8.0)

    grad_owner = object.__new__(FixedBoundaryExactOptimizer)
    grad_owner.objective_and_gradient_fun = lambda params: (1.0, np.asarray(params, dtype=float) + 2.0)
    np.testing.assert_allclose(FixedBoundaryExactOptimizer.gradient_fun(grad_owner, np.asarray([3.0])), [5.0])


def test_empty_tangent_columns_and_cached_linear_operator_matmat(monkeypatch) -> None:
    import jax.numpy as jnp
    import vmec_jax.discrete_adjoint as adjoint_module

    state = _state_from_coeffs(r=1.0, rs=2.0)
    empty_owner = object.__new__(FixedBoundaryExactOptimizer)
    empty_owner._layout = state.layout
    empty_owner._solve_exact_with_tape = lambda params, *, return_payload: (state, {"tape": None, "axis_override": {}})
    empty_state, empty_tangents = empty_owner._state_and_tangent_columns(np.asarray([], dtype=float), profile_prefix="empty")
    assert empty_state is state
    assert np.asarray(empty_tangents).shape == (0, state.layout.size)
    np.testing.assert_allclose(
        np.asarray(empty_owner._initial_tangent_columns(np.asarray([], dtype=float), {}, profile_prefix="empty")),
        np.zeros((0, state.layout.size)),
    )

    monkeypatch.setattr(adjoint_module, "checkpoint_tape_state_jvp_columns", lambda **kwargs: kwargs["initial_tangents"])

    lin_owner = object.__new__(FixedBoundaryExactOptimizer)
    lin_owner._solver_device_name = None
    lin_owner._profile = {}
    lin_owner._static = SimpleNamespace(cfg=SimpleNamespace(lasym=False))
    lin_owner._layout = state.layout
    lin_owner._indata = object()
    lin_owner._signgs = 1
    lin_owner._discrete_jacobian_helper_cache = {}
    lin_owner._initial_tangent_cache = {
        "cached": jnp.asarray(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=jnp.float64,
        )
    }
    lin_owner._initial_tangent_cache_key = lambda _params: "cached"
    lin_owner._exact_cache_key = lambda params: np.asarray(params, dtype=float).reshape(-1).tobytes()
    lin_owner._exact_residual_cache = {}
    lin_owner._solve_exact_with_tape = lambda params, *, return_payload: (state, {"tape": "tape", "axis_override": {}})
    lin_owner._residuals_fn = lambda state_arg: jnp.asarray(
        [
            state_arg.Rcos[0, 0] + state_arg.Rsin[0, 0],
            3.0 * state_arg.Rcos[0, 0] - state_arg.Rsin[0, 0],
        ],
        dtype=jnp.float64,
    )

    operator = lin_owner.residual_linear_operator(np.asarray([0.0, 0.0]))

    np.testing.assert_allclose(
        operator.matmat(np.asarray([[1.0, 2.0], [3.0, 4.0]])),
        np.asarray([[4.0, 6.0], [0.0, 2.0]]),
    )
    assert lin_owner._profile["linear_operator_initial_tangents_cache_hit"]["count"] == 1


def test_jacobian_tracking_exact_residual_after_jacobian_and_save_wout_cache(monkeypatch, tmp_path) -> None:
    import vmec_jax.driver as driver_module

    owner = object.__new__(FixedBoundaryExactOptimizer)
    owner._exact_cache_key = lambda _params: b"point"
    owner._last_jacobian_key = [None]
    owner._last_jacobian_residual = None
    owner._last_jacobian_source = "exact_tape_replay"
    owner._history = []
    owner._wall_t0 = opt_module.time.perf_counter()
    owner._exact_state_cache = {b"point": "scan-state"}
    owner._exact_cache = {}
    owner._scan_exact_path = "scan"
    owner._can_build_history_from_residuals = lambda: False
    owner._cached_exact_residual = lambda cache_key=None, **_kwargs: np.asarray([1.0, 2.0]) if cache_key == b"point" else None
    owner._history_entry_from_state_or_residual = lambda state, res, **kwargs: {
        "cost": 0.5,
        "state": state,
        "residual_sum": float(np.sum(res)),
    }
    owner._exact_history_accepts = lambda cost: bool(cost <= 1.0)
    owner._remember_best_exact_point = lambda params, residual, cost=None, **kwargs: setattr(
        owner, "_best_seen", (params, residual, cost, kwargs.get("state"))
    )
    owner.jacobian_fun = lambda _params: np.asarray([[7.0]])

    np.testing.assert_allclose(owner._jacobian_fun_tracked(np.asarray([0.0])), [[7.0]])
    assert owner._history[-1]["state"] == "scan-state"
    assert owner._history[-1]["residual_sum"] == pytest.approx(3.0)

    owner._exact_state_cache = {}
    owner._exact_cache = {b"point": ("cached-state", {})}
    np.testing.assert_allclose(owner._jacobian_fun_tracked(np.asarray([0.0])), [[7.0]])
    assert owner._history[-1]["state"] == "cached-state"

    owner._last_jacobian_key = [None]
    owner._last_jacobian_residual = np.asarray([3.0])
    np.testing.assert_allclose(owner._exact_residual_after_jacobian(), [3.0])
    owner._last_jacobian_residual = None
    owner._exact_cache = {}
    assert owner._exact_residual_after_jacobian() is None

    saved_states = []
    monkeypatch.setattr(driver_module, "write_wout_from_fixed_boundary_run", lambda path, run, **_kwargs: saved_states.append(run.state))

    save_owner = object.__new__(FixedBoundaryExactOptimizer)
    save_owner._state_matches_params = lambda state, params: False
    save_owner._cached_exact_state = lambda params: "cached-save-state"
    save_owner._solve_forward = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected solve"))
    save_owner._static = SimpleNamespace(cfg="cfg")
    save_owner._indata = "indata"
    save_owner._flux = "flux"
    save_owner._signgs = -1
    save_owner._profile = {}

    save_owner.save_wout(tmp_path / "wout_cached.nc", params=np.asarray([0.0]), state="stale-state")

    assert saved_states == ["cached-save-state"]
