from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

import vmec_jax.optimization as opt_module
from vmec_jax.optimization import (
    BoundaryParamSpec,
    FixedBoundaryExactOptimizer,
    _linear_operator_matrix_arg,
    _linear_operator_vector_arg,
)
from vmec_jax.state import StateLayout, VMECState, pack_state


def _state_from_coeffs(r=0.0, rs=0.0, z=0.0, zs=0.0, l=0.0, ls=0.0) -> VMECState:
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


def _bare_optimizer_for_state_ops() -> FixedBoundaryExactOptimizer:
    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._solver_device_name = None
    opt._inside_solver_device_context = False
    opt._static = SimpleNamespace(cfg=SimpleNamespace(lasym=False), grid=SimpleNamespace(ntheta=1, nzeta=1))
    opt._indata = object()
    opt._signgs = -1
    opt._layout = StateLayout(ns=1, K=1, lasym=False)
    opt._profile = {}
    opt._exact_cache = {}
    opt._exact_state_cache = {}
    opt._exact_residual_cache = {}
    opt._discrete_jacobian_helper_cache = {}
    opt._initial_tangent_cache = {}
    opt._exact_cache_key = lambda params: np.asarray(params, dtype=float).reshape(-1).tobytes()
    opt._remember_exact_residual = lambda key, residual: opt._exact_residual_cache.update(
        {key: np.asarray(residual, dtype=float)}
    )
    opt._profile_add = FixedBoundaryExactOptimizer._profile_add.__get__(opt, FixedBoundaryExactOptimizer)
    opt._lasym_replay_column_chunk = lambda _n_params: None
    return opt


def test_linear_operator_arg_helpers_validate_vectors_and_matrix_edges() -> None:
    np.testing.assert_allclose(_linear_operator_vector_arg([[1.0], [2.0]], size=2, name="v"), [1.0, 2.0])
    with pytest.raises(ValueError, match="bad expected 3 entries"):
        _linear_operator_vector_arg([1.0, 2.0], size=3, name="bad")

    np.testing.assert_allclose(
        _linear_operator_matrix_arg(np.asarray([1.0, 2.0, 3.0, 4.0]), rows=2, name="m"),
        [[1.0, 2.0], [3.0, 4.0]],
    )
    assert _linear_operator_matrix_arg([], rows=0, name="empty").shape == (0, 0)
    with pytest.raises(ValueError, match="zero expected 0 rows"):
        _linear_operator_matrix_arg([1.0], rows=0, name="zero")
    with pytest.raises(ValueError, match="cannot be reshaped"):
        _linear_operator_matrix_arg([1.0, 2.0, 3.0], rows=2, name="ragged")
    with pytest.raises(ValueError, match="rows expected 3 rows"):
        _linear_operator_matrix_arg(np.ones((2, 2)), rows=3, name="rows")


def test_initial_state_from_params_uses_jit_helper_and_cache(monkeypatch) -> None:
    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._initial_state_cache = OrderedDict()
    opt._initial_state_cache_max = 4
    opt._profile = {}
    opt._exact_cache_key = lambda params: np.asarray(params, dtype=float).reshape(-1).tobytes()
    opt._profile_add = FixedBoundaryExactOptimizer._profile_add.__get__(opt, FixedBoundaryExactOptimizer)
    opt._remember_initial_state = FixedBoundaryExactOptimizer._remember_initial_state.__get__(
        opt,
        FixedBoundaryExactOptimizer,
    )
    state = _state_from_coeffs(r=2.0)
    calls = {"jit": 0}

    def fake_jit(params):
        calls["jit"] += 1
        return state

    opt._initial_state_from_params_jit = fake_jit
    monkeypatch.setattr(
        opt_module,
        "initial_guess_from_boundary",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("fallback should not run")),
    )

    params = np.asarray([1.0, 2.0])
    assert FixedBoundaryExactOptimizer._initial_state_from_params(opt, params, profile_name="initial") is state
    assert calls["jit"] == 1
    assert opt._profile["initial"]["count"] == 1

    assert FixedBoundaryExactOptimizer._initial_state_from_params(opt, params, profile_name="initial") is state
    assert calls["jit"] == 1
    assert opt._profile["initial_cache_hit"]["count"] == 1


def test_initial_state_from_params_falls_back_when_jit_disabled(monkeypatch) -> None:
    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._initial_state_cache = OrderedDict()
    opt._initial_state_cache_max = 4
    opt._profile = {}
    opt._static = object()
    opt._indata = object()
    opt._exact_cache_key = lambda params: np.asarray(params, dtype=float).reshape(-1).tobytes()
    opt._profile_add = FixedBoundaryExactOptimizer._profile_add.__get__(opt, FixedBoundaryExactOptimizer)
    opt._remember_initial_state = FixedBoundaryExactOptimizer._remember_initial_state.__get__(
        opt,
        FixedBoundaryExactOptimizer,
    )
    opt._initial_state_from_params_jit = lambda _params: None
    opt._boundary_from_params = lambda params: ("boundary", tuple(np.asarray(params, dtype=float)))
    state = _state_from_coeffs(r=3.0)
    seen = []

    def fake_initial_guess(static, boundary, indata, *, vmec_project):
        seen.append((static, boundary, indata, vmec_project))
        return state

    monkeypatch.setattr(opt_module, "initial_guess_from_boundary", fake_initial_guess)

    params = np.asarray([4.0])
    assert FixedBoundaryExactOptimizer._initial_state_from_params(opt, params, profile_name="initial") is state
    assert seen == [(opt._static, ("boundary", (4.0,)), opt._indata, True)]
    assert opt._profile["initial"]["count"] == 1


def test_jit_initial_state_env_and_clear_caches(monkeypatch) -> None:
    opt = _bare_optimizer_for_state_ops()
    opt._initial_state_packed_helper = object()
    opt._exact_jacobian_cache = {}
    opt._trial_residual_cache = OrderedDict()
    opt._initial_state_cache = OrderedDict()
    opt._initial_tangent_cache = {}
    opt._last_jacobian_residual = np.asarray([1.0])
    opt._post_jacobian_clear = lambda *, clear_compiled=False: None

    monkeypatch.delenv("VMEC_JAX_OPT_JIT_INITIAL_STATE", raising=False)
    assert FixedBoundaryExactOptimizer._use_jit_initial_state(opt) is True
    opt._solver_device_name = "gpu"
    assert FixedBoundaryExactOptimizer._use_jit_initial_state(opt) is False
    monkeypatch.setenv("VMEC_JAX_OPT_JIT_INITIAL_STATE", "0")
    assert FixedBoundaryExactOptimizer._use_jit_initial_state(opt) is False
    monkeypatch.setenv("VMEC_JAX_OPT_JIT_INITIAL_STATE", "1")
    assert FixedBoundaryExactOptimizer._use_jit_initial_state(opt) is True

    FixedBoundaryExactOptimizer.clear_caches(opt)
    assert opt._initial_state_packed_helper is None
    assert opt._last_jacobian_residual is None


def test_solver_device_context_and_trial_scan_env_branches(monkeypatch) -> None:
    import vmec_jax._compat as compat

    events: list[tuple[str, object]] = []

    @contextmanager
    def fake_default_device(device):
        events.append(("enter", device))
        yield
        events.append(("exit", device))

    fake_jax = SimpleNamespace(
        devices=lambda name: [f"{name}:0"],
        default_device=fake_default_device,
        default_backend=lambda: "gpu",
    )
    monkeypatch.setattr(compat, "jax", fake_jax)

    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._solver_device_name = "gpu"
    opt._inside_solver_device_context = False
    with opt._solver_device_context():
        events.append(("body", "ok"))
    assert events == [("enter", "gpu:0"), ("body", "ok"), ("exit", "gpu:0")]

    opt._solver_device_name = None
    assert isinstance(opt._solver_device_context(), nullcontext)

    opt._solver_device_name = "missing"
    monkeypatch.setattr(fake_jax, "devices", lambda _name: [])
    assert isinstance(opt._solver_device_context(), nullcontext)

    opt._inside_solver_device_context = False
    opt._solver_device_name = "cpu"
    seen = []
    monkeypatch.setattr(
        FixedBoundaryExactOptimizer,
        "_solver_device_context",
        lambda self: fake_default_device("cpu:0"),
    )
    assert opt._run_in_solver_device_context(lambda value: seen.append(value) or "done", 5) == "done"
    assert opt._inside_solver_device_context is False
    assert seen == [5]

    monkeypatch.setenv("VMEC_JAX_OPT_EXACT_PATH", "scan")
    assert opt._select_exact_path() == "scan"
    monkeypatch.setenv("VMEC_JAX_OPT_EXACT_PATH", "tape")
    assert opt._select_exact_path() == "tape"
    monkeypatch.delenv("VMEC_JAX_OPT_EXACT_PATH")
    assert opt._select_exact_path() == "tape"

    monkeypatch.setenv("VMEC_JAX_OPT_TRIAL_SCAN", "0")
    assert opt._use_scan_for_trial_solves() is False
    monkeypatch.setenv("VMEC_JAX_OPT_TRIAL_SCAN", "scan")
    assert opt._use_scan_for_trial_solves() is True
    monkeypatch.delenv("VMEC_JAX_OPT_TRIAL_SCAN")
    opt._solver_device_name = "gpu"
    assert opt._use_scan_for_trial_solves() is True
    opt._solver_device_name = "cpu"
    assert opt._use_scan_for_trial_solves() is False


def test_exact_tape_precomputed_tridi_policy_backend_and_env(monkeypatch) -> None:
    import vmec_jax._compat as compat

    opt = object.__new__(FixedBoundaryExactOptimizer)

    opt._solver_device_name = "gpu"
    opt._specs = [object()] * 8
    monkeypatch.delenv("VMEC_JAX_OPT_EXACT_TRIDI_PRECOMPUTE", raising=False)
    monkeypatch.delenv("VMEC_JAX_OPT_EXACT_TRIDI_PRECOMPUTE_MAX_DOFS", raising=False)
    assert opt._use_precomputed_tridi_for_exact_tape() is True
    opt._specs = [object()] * 24
    assert opt._use_precomputed_tridi_for_exact_tape() is None
    monkeypatch.setenv("VMEC_JAX_OPT_EXACT_TRIDI_PRECOMPUTE_MAX_DOFS", "24")
    assert opt._use_precomputed_tridi_for_exact_tape() is True
    monkeypatch.setenv("VMEC_JAX_OPT_EXACT_TRIDI_PRECOMPUTE_MAX_DOFS", "-1")
    assert opt._use_precomputed_tridi_for_exact_tape() is False
    monkeypatch.delenv("VMEC_JAX_OPT_EXACT_TRIDI_PRECOMPUTE_MAX_DOFS")

    opt._solver_device_name = "cpu"
    assert opt._use_precomputed_tridi_for_exact_tape() is None

    monkeypatch.setenv("VMEC_JAX_OPT_EXACT_TRIDI_PRECOMPUTE", "0")
    assert opt._use_precomputed_tridi_for_exact_tape() is False
    monkeypatch.setenv("VMEC_JAX_OPT_EXACT_TRIDI_PRECOMPUTE", "yes")
    assert opt._use_precomputed_tridi_for_exact_tape() is True

    monkeypatch.delenv("VMEC_JAX_OPT_EXACT_TRIDI_PRECOMPUTE")
    opt._solver_device_name = None
    opt._specs = [object()] * 8
    monkeypatch.setattr(compat, "jax", SimpleNamespace(default_backend=lambda: "cuda"))
    assert opt._use_precomputed_tridi_for_exact_tape() is True
    opt._specs = [object()] * 24
    assert opt._use_precomputed_tridi_for_exact_tape() is None
    monkeypatch.setattr(compat, "jax", SimpleNamespace(default_backend=lambda: "cpu"))
    assert opt._use_precomputed_tridi_for_exact_tape() is None


def test_optimizer_init_moves_static_boundary_and_boundary_input(monkeypatch) -> None:
    state0 = SimpleNamespace(layout=SimpleNamespace(size=6))
    static = SimpleNamespace(s=np.asarray([0.0, 1.0]), cfg=SimpleNamespace(lasym=False))
    indata = SimpleNamespace(
        get=lambda key, default=None: default,
        get_float=lambda key, default=0.0: default,
    )
    boundary = object()
    boundary_input = object()
    residuals_fn = lambda _state: np.asarray([1.0])
    moved = []

    def fake_move(self, value):
        moved.append(value)
        return value

    # The optimizer now no-ops explicit device selection if that backend is
    # already active.  Make the active backend different here so this test
    # still covers the recursive static/boundary transfer path.
    monkeypatch.setattr("vmec_jax._compat.jax", SimpleNamespace(default_backend=lambda: "gpu"))
    monkeypatch.setattr(FixedBoundaryExactOptimizer, "_move_to_solver_device", fake_move)
    monkeypatch.setattr(FixedBoundaryExactOptimizer, "_make_residuals_eval_fn", lambda self, fn: fn)
    monkeypatch.setattr(opt_module, "initial_guess_from_boundary", lambda *args, **kwargs: state0)
    monkeypatch.setattr(opt_module, "eval_geom", lambda *_args, **_kwargs: SimpleNamespace(sqrtg=np.ones((2, 2))))
    monkeypatch.setattr(opt_module, "signgs_from_sqrtg", lambda *_args, **_kwargs: -1)
    monkeypatch.setattr(opt_module, "flux_profiles_from_indata", lambda *_args, **_kwargs: "flux")

    opt = FixedBoundaryExactOptimizer(
        static,
        indata,
        boundary,
        [BoundaryParamSpec("rc10", "rc", 0, 1, 0)],
        residuals_fn,
        boundary_input=boundary_input,
        solver_device="cpu",
    )

    assert moved == [static, boundary, boundary_input]
    assert opt.static is static
    assert opt.flux == "flux"
    assert opt._exact_solver_kwargs["preconditioner_use_precomputed_tridi"] is None
    assert "preconditioner_use_precomputed_tridi" not in opt._trial_solver_kwargs


def test_lasym_replay_column_chunk_env_and_backend_branches(monkeypatch) -> None:
    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._static = SimpleNamespace(cfg=SimpleNamespace(lasym=True))
    opt._solver_device_name = "gpu"

    monkeypatch.setenv("VMEC_JAX_LASYM_REPLAY_COLUMN_CHUNK", "3")
    assert opt._lasym_replay_column_chunk(12) == 3
    monkeypatch.setenv("VMEC_JAX_LASYM_REPLAY_COLUMN_CHUNK", "0")
    assert opt._lasym_replay_column_chunk(12) is None
    monkeypatch.setenv("VMEC_JAX_LASYM_REPLAY_COLUMN_CHUNK", "bad")
    assert opt._lasym_replay_column_chunk(24) is None
    monkeypatch.delenv("VMEC_JAX_LASYM_REPLAY_COLUMN_CHUNK")

    monkeypatch.setenv("VMEC_JAX_REPLAY_COLUMN_CHUNK", "anything")
    assert opt._lasym_replay_column_chunk(128) is None
    monkeypatch.delenv("VMEC_JAX_REPLAY_COLUMN_CHUNK")

    assert opt._lasym_replay_column_chunk(23) is None
    assert opt._lasym_replay_column_chunk(24) is None
    assert opt._lasym_replay_column_chunk(96) == 24

    opt._solver_device_name = "tpu"
    assert opt._lasym_replay_column_chunk(128) is None

    opt._solver_device_name = "cpu"
    assert opt._lasym_replay_column_chunk(31) is None
    assert opt._lasym_replay_column_chunk(32) == 4
    assert opt._lasym_replay_column_chunk(64) == 8

    opt._static = SimpleNamespace(cfg=SimpleNamespace(lasym=False))
    assert opt._lasym_replay_column_chunk(128) is None

    opt._solver_device_name = "gpu"
    assert opt._lasym_replay_column_chunk(24) is None
    assert opt._lasym_replay_column_chunk(48) == 24


def test_projected_replay_residuals_env_and_backend_branches(monkeypatch) -> None:
    import vmec_jax._compat as compat

    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._solver_device_name = "gpu"

    monkeypatch.setenv("VMEC_JAX_OPT_PROJECTED_REPLAY_RESIDUALS", "0")
    assert opt._projected_replay_residuals_enabled() is False
    monkeypatch.setenv("VMEC_JAX_OPT_PROJECTED_REPLAY_RESIDUALS", "yes")
    assert opt._projected_replay_residuals_enabled() is True
    monkeypatch.delenv("VMEC_JAX_OPT_PROJECTED_REPLAY_RESIDUALS")

    opt._static = SimpleNamespace(cfg=SimpleNamespace(lasym=False))
    assert opt._projected_replay_residuals_enabled(24) is True
    assert opt._projected_replay_residuals_enabled(48) is True

    opt._static = SimpleNamespace(cfg=SimpleNamespace(lasym=True))
    assert opt._projected_replay_residuals_enabled(48) is False

    opt._static = SimpleNamespace(cfg=SimpleNamespace(lasym=False))
    opt._solver_device_name = "cpu"
    assert opt._projected_replay_residuals_enabled(48) is False

    opt._solver_device_name = None
    monkeypatch.setattr(compat, "jax", SimpleNamespace(default_backend=lambda: "cuda"))
    assert opt._projected_replay_residuals_enabled(48) is True
    assert opt._projected_replay_residuals_enabled(24) is True
    monkeypatch.setattr(compat, "jax", SimpleNamespace(default_backend=lambda: "cpu"))
    assert opt._projected_replay_residuals_enabled(48) is False
    monkeypatch.setattr(
        compat,
        "jax",
        SimpleNamespace(default_backend=lambda: (_ for _ in ()).throw(RuntimeError("backend probe failed"))),
    )
    assert opt._projected_replay_residuals_enabled(48) is False


def test_fused_projected_replay_is_opt_in(monkeypatch) -> None:
    opt = object.__new__(FixedBoundaryExactOptimizer)

    monkeypatch.delenv("VMEC_JAX_OPT_FUSED_PROJECTED_REPLAY", raising=False)
    assert opt._fused_projected_replay_enabled() is False

    monkeypatch.setenv("VMEC_JAX_OPT_FUSED_PROJECTED_REPLAY", "1")
    assert opt._fused_projected_replay_enabled() is True

    monkeypatch.setenv("VMEC_JAX_OPT_FUSED_PROJECTED_REPLAY", "no")
    assert opt._fused_projected_replay_enabled() is False


def test_projected_replay_jacobian_path_projects_without_intermediate_sync(monkeypatch) -> None:
    import jax.numpy as jnp
    import vmec_jax.discrete_adjoint as adjoint_module

    state = _state_from_coeffs(r=1.0, rs=2.0, z=3.0, zs=4.0, l=5.0, ls=6.0)
    tangents = jnp.asarray(
        [
            np.arange(state.layout.size, dtype=float) + 1.0,
            np.arange(state.layout.size, dtype=float) + 10.0,
        ],
        dtype=jnp.float64,
    )
    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._layout = state.layout
    opt._static = SimpleNamespace(cfg=SimpleNamespace(lasym=False))
    opt._profile = {}
    opt._discrete_jacobian_helper_cache = {}
    opt._solve_exact_with_tape_for_jvp = lambda _params: (
        state,
        {"tape": "tape", "axis_override": {"axis": "unused"}},
    )
    opt._initial_tangent_columns = lambda _params, _axis_override, *, profile_prefix: tangents
    opt._lasym_replay_column_chunk = lambda _n_params: None
    opt._residuals_fn = lambda state_arg: pack_state(state_arg)[:2]
    remembered = {}
    opt._remember_exact_residual = lambda key, residual: remembered.update({"residual": (key, residual)})
    opt._remember_exact_jacobian = lambda key, jac, residual: remembered.update({"jacobian": (key, jac, residual)})
    monkeypatch.setattr(
        adjoint_module,
        "checkpoint_tape_state_jvp_columns",
        lambda **kwargs: kwargs["initial_tangents"],
    )

    jac = opt._jacobian_fun_projected_replay(
        jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        b"exact-key",
        t_total=opt_module.time.perf_counter(),
    )

    np.testing.assert_allclose(jac, [[1.0, 10.0], [2.0, 11.0]])
    np.testing.assert_allclose(opt._last_jacobian_residual, np.asarray(pack_state(state)[:2]))
    assert opt._last_jacobian_source == "exact_tape_projected_replay"
    assert remembered["residual"][0] == b"exact-key"
    assert remembered["jacobian"][0] == b"exact-key"
    assert opt._profile["jacobian_projected_tape_replay_dispatch"]["count"] == 1
    assert opt._profile["jacobian_projected_replay_residual_tangents"]["count"] == 1


def test_projected_replay_jacobian_path_can_fuse_dynamic_basepoint(monkeypatch) -> None:
    import jax.numpy as jnp
    import vmec_jax.discrete_adjoint as adjoint_module

    monkeypatch.setenv("VMEC_JAX_OPT_FUSED_PROJECTED_REPLAY", "1")
    state = _state_from_coeffs(r=1.0, rs=2.0, z=3.0, zs=4.0, l=5.0, ls=6.0)
    layout_size = int(state.layout.size)
    tangents = jnp.asarray(
        [
            np.arange(layout_size, dtype=float) + 1.0,
            np.arange(layout_size, dtype=float) + 10.0,
        ],
        dtype=jnp.float64,
    )
    stacked = {"active": jnp.asarray([True])}
    stacked_base_carries = (jnp.zeros((1, layout_size), dtype=jnp.float64),) + tuple(
        jnp.zeros((1, 1), dtype=jnp.float64) for _ in range(14)
    )
    static_flags = {
        "apply_lforbal": False,
        "include_edge_residual": True,
        "apply_m1_constraints": True,
        "limit_update_rms": False,
        "limit_dt_from_force": False,
        "vmec2000_control": True,
        "divide_by_scalxc_for_update": False,
        "signgs": 1,
        "precond_jmax": 1,
    }
    tape = SimpleNamespace(
        stacked_step_traces=stacked,
        dynamic_base_carries_stacked=stacked_base_carries,
        step_trace_static_flags=static_flags,
    )
    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._layout = state.layout
    opt._static = SimpleNamespace(cfg=SimpleNamespace(lasym=False))
    opt._profile = {}
    opt._discrete_jacobian_helper_cache = {}
    opt._solve_exact_with_tape_for_jvp = lambda _params: (
        state,
        {"tape": tape, "axis_override": {"axis": "unused"}},
    )
    opt._initial_tangent_columns = lambda _params, _axis_override, *, profile_prefix: tangents
    opt._lasym_replay_column_chunk = lambda _n_params: None
    opt._residuals_fn = lambda state_arg: pack_state(state_arg)[:2]
    remembered = {}
    opt._remember_exact_residual = lambda key, residual: remembered.update({"residual": (key, residual)})
    opt._remember_exact_jacobian = lambda key, jac, residual: remembered.update({"jacobian": (key, jac, residual)})

    def fake_runner(*, static, stacked, stacked_base_carries, static_flags):
        def run_scan(carry_tangents0, _stacked_base_carries_in, _stacked_traces_in):
            return (carry_tangents0[0] + 2.0,) + tuple(carry_tangents0[1:])

        return run_scan

    monkeypatch.setattr(adjoint_module, "_checkpoint_tape_dynamic_basepoint_scan_runner", fake_runner)
    monkeypatch.setattr(adjoint_module, "_dynamic_basepoint_payload_shapes_match", lambda *_args: True)
    monkeypatch.setattr(adjoint_module, "_stacked_trace_signature", lambda _tree: ("sig",))

    jac = opt._jacobian_fun_projected_replay(
        jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        b"exact-key",
        t_total=opt_module.time.perf_counter(),
    )

    np.testing.assert_allclose(jac, [[3.0, 12.0], [4.0, 13.0]])
    np.testing.assert_allclose(opt._last_jacobian_residual, np.asarray(pack_state(state)[:2]))
    assert opt._last_jacobian_source == "exact_tape_fused_projected_replay"
    assert remembered["residual"][0] == b"exact-key"
    assert remembered["jacobian"][0] == b"exact-key"
    assert opt._profile["jacobian_fused_projected_replay_total"]["count"] == 1
    assert "jacobian_projected_tape_replay_dispatch" not in opt._profile
    assert "jacobian_residual_tangent_helper_build" not in opt._profile
    assert all(
        "stacked" not in cache and "stacked_base_carries" not in cache
        for cache in opt._discrete_jacobian_helper_cache.values()
    )


def test_projected_replay_fused_path_respects_explicit_column_chunk(monkeypatch) -> None:
    import jax.numpy as jnp
    import vmec_jax.discrete_adjoint as adjoint_module

    monkeypatch.setenv("VMEC_JAX_OPT_FUSED_PROJECTED_REPLAY", "1")
    monkeypatch.setenv("VMEC_JAX_REPLAY_COLUMN_CHUNK", "2")
    state = _state_from_coeffs(r=1.0, rs=2.0, z=3.0, zs=4.0, l=5.0, ls=6.0)
    layout_size = int(state.layout.size)
    tangents = jnp.asarray(
        [
            np.arange(layout_size, dtype=float) + 1.0,
            np.arange(layout_size, dtype=float) + 10.0,
            np.arange(layout_size, dtype=float) + 20.0,
        ],
        dtype=jnp.float64,
    )
    tape = SimpleNamespace(
        stacked_step_traces={"active": jnp.asarray([True])},
        dynamic_base_carries_stacked=(jnp.zeros((1, layout_size), dtype=jnp.float64),)
        + tuple(jnp.zeros((1, 1), dtype=jnp.float64) for _ in range(14)),
        step_trace_static_flags={
            "apply_lforbal": False,
            "include_edge_residual": True,
            "apply_m1_constraints": True,
            "limit_update_rms": False,
            "limit_dt_from_force": False,
            "vmec2000_control": True,
            "divide_by_scalxc_for_update": False,
            "signgs": 1,
            "precond_jmax": 1,
        },
    )
    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._layout = state.layout
    opt._static = SimpleNamespace(cfg=SimpleNamespace(lasym=False))
    opt._profile = {}
    opt._discrete_jacobian_helper_cache = {}
    opt._solve_exact_with_tape_for_jvp = lambda _params: (
        state,
        {"tape": tape, "axis_override": {"axis": "unused"}},
    )
    opt._initial_tangent_columns = lambda _params, _axis_override, *, profile_prefix: tangents
    opt._lasym_replay_column_chunk = lambda _n_params: None
    opt._residuals_fn = lambda state_arg: pack_state(state_arg)[:2]
    opt._remember_exact_residual = lambda _key, _residual: None
    opt._remember_exact_jacobian = lambda _key, _jac, _residual: None
    monkeypatch.setattr(adjoint_module, "_dynamic_basepoint_payload_shapes_match", lambda *_args: True)
    monkeypatch.setattr(adjoint_module, "_stacked_trace_signature", lambda _tree: ("sig",))
    monkeypatch.setattr(
        adjoint_module,
        "checkpoint_tape_state_jvp_columns",
        lambda **kwargs: kwargs["initial_tangents"] + 5.0,
    )

    jac = opt._jacobian_fun_projected_replay(
        jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float64),
        b"exact-key",
        t_total=opt_module.time.perf_counter(),
    )

    np.testing.assert_allclose(jac, [[6.0, 15.0, 25.0], [7.0, 16.0, 26.0]])
    assert opt._last_jacobian_source == "exact_tape_projected_replay"
    assert "jacobian_fused_projected_replay_total" not in opt._profile
    assert opt._profile["jacobian_projected_tape_replay_dispatch"]["count"] == 1


def test_cached_exact_residual_none_and_lasym_backend_probe_failure(monkeypatch) -> None:
    import vmec_jax._compat as compat

    opt = _bare_optimizer_for_state_ops()
    assert opt._cached_exact_residual() is None

    opt._static = SimpleNamespace(cfg=SimpleNamespace(lasym=True))
    opt._solver_device_name = None
    monkeypatch.delenv("VMEC_JAX_LASYM_REPLAY_COLUMN_CHUNK", raising=False)
    monkeypatch.delenv("VMEC_JAX_REPLAY_COLUMN_CHUNK", raising=False)
    monkeypatch.setattr(
        compat,
        "jax",
        SimpleNamespace(default_backend=lambda: (_ for _ in ()).throw(RuntimeError("backend unavailable"))),
    )

    assert opt._lasym_replay_column_chunk(128) is None


def test_scan_jacobian_path_records_residual_and_optionally_solves_state() -> None:
    opt = _bare_optimizer_for_state_ops()
    opt._scan_exact_path = "scan"
    opt._can_build_history_from_residuals = lambda: False
    solved = []
    opt._solve_scan_exact_state = lambda params: solved.append(np.asarray(params, dtype=float).copy()) or "state"
    opt._scan_exact_helpers = lambda: {
        "residual_and_jacobian": lambda params: (
            np.asarray([1.0, 2.0]),
            np.asarray([[3.0, 4.0], [5.0, 6.0]]),
        )
    }

    jac = opt.jacobian_fun(np.asarray([0.25, 0.5]))

    np.testing.assert_allclose(jac, [[3.0, 4.0], [5.0, 6.0]])
    np.testing.assert_allclose(opt._last_jacobian_residual, [1.0, 2.0])
    assert len(solved) == 1
    assert opt._profile["scan_jacobian_total"]["count"] == 1


def test_scan_exact_helpers_build_cached_state_residual_and_jacobian(monkeypatch) -> None:
    import jax.numpy as jnp
    import vmec_jax.solve as solve_module

    opt = _bare_optimizer_for_state_ops()
    opt._specs = [object(), object()]
    opt._inner_max_iter = 2
    opt._inner_ftol = 1.0e-6
    opt._solver_device_name = None
    opt._exact_solver_kwargs = {"marker": True}
    opt._scan_exact_helper_cache = {}
    opt._boundary_from_params = lambda params: params

    def fake_initial_guess(_static, boundary, _indata, *, vmec_project):
        assert vmec_project is True
        return _state_from_coeffs(r=boundary[0], rs=boundary[1])

    def fake_solve(state0, static, **kwargs):
        assert static is opt._static
        assert kwargs["max_iter"] == 2
        assert kwargs["ftol"] == pytest.approx(1.0e-6)
        assert kwargs["use_scan"] is True
        assert kwargs["state_only"] is True
        assert kwargs["resume_state_mode"] == "none"
        return SimpleNamespace(state=state0)

    monkeypatch.setattr(opt_module, "initial_guess_from_boundary", fake_initial_guess)
    monkeypatch.setattr(solve_module, "solve_fixed_boundary_residual_iter", fake_solve)
    opt._residuals_fn = lambda state: jnp.asarray([state.Rcos[0, 0] + 2.0 * state.Rsin[0, 0]], dtype=jnp.float64)

    helpers = opt._scan_exact_helpers()
    residual = helpers["residual"](jnp.asarray([1.0, 2.0], dtype=jnp.float64))
    residual2, jac = helpers["residual_and_jacobian"](jnp.asarray([1.0, 2.0], dtype=jnp.float64))

    np.testing.assert_allclose(np.asarray(residual), [5.0])
    np.testing.assert_allclose(np.asarray(residual2), [5.0])
    np.testing.assert_allclose(np.asarray(jac), [[1.0, 2.0]])
    assert opt._scan_exact_helpers() is helpers


def test_tangent_and_b_cartesian_helpers_cover_zero_and_nonzero_columns(monkeypatch) -> None:
    import jax.numpy as jnp
    import vmec_jax.field as field_module

    opt = _bare_optimizer_for_state_ops()
    state = _state_from_coeffs(r=1.0, z=2.0, l=3.0)

    opt._state_and_tangent_columns = lambda params, *, profile_prefix: (
        state,
        np.zeros((int(np.asarray(params).size), int(state.layout.size))),
    )
    state_out, tangents = opt.state_tangent_columns_fun(np.asarray([1.0, 2.0]))
    assert state_out is state
    assert tangents.shape == (2, state.layout.size)
    assert opt._profile["state_tangent_columns_total"]["count"] == 1

    def fake_b_cartesian_from_state(state_arg, static, *, indata, signgs, s_index):
        assert (static, indata, signgs, s_index) == (opt._static, opt._indata, -1, -2)
        return jnp.asarray([state_arg.Rcos[0, 0], state_arg.Zcos[0, 0], state_arg.Lcos[0, 0]]).reshape((1, 1, 3))

    monkeypatch.setattr(field_module, "b_cartesian_from_state", fake_b_cartesian_from_state)

    field, tangent_columns = opt.b_cartesian_tangent_columns_fun(np.asarray([], dtype=float), s_index=-2)

    np.testing.assert_allclose(field, [[[1.0, 2.0, 3.0]]])
    assert tangent_columns.shape == (1, 1, 3, 0)


def test_objective_and_gradient_uses_residual_vjp_and_cotangent_factory(monkeypatch) -> None:
    import jax.numpy as jnp
    import vmec_jax.discrete_adjoint as adjoint_module
    import vmec_jax.init_guess as init_guess_module

    opt = _bare_optimizer_for_state_ops()
    opt._initial_tangent_cache_key = lambda params: ("test-key", np.asarray(params, dtype=float).shape)
    state = _state_from_coeffs(r=2.0, z=3.0)
    opt._solve_exact_with_tape = lambda _params, *, return_payload: (
        state,
        {"tape": "tape", "axis_override": {}},
    )
    opt._boundary_from_params = lambda params: params
    monkeypatch.setattr(
        init_guess_module,
        "initial_guess_from_boundary",
        lambda _static, boundary, _indata, **_kwargs: _state_from_coeffs(r=boundary[0], z=boundary[1]),
    )
    monkeypatch.setattr(adjoint_module, "checkpoint_tape_state_vjp", lambda **kwargs: kwargs["final_cotangent"])

    def residuals_fn(state_arg):
        return jnp.asarray([state_arg.Rcos[0, 0] + 2.0 * state_arg.Zcos[0, 0]], dtype=jnp.float64)

    opt._residuals_fn = residuals_fn
    cost, grad = opt.objective_and_gradient_fun(np.asarray([0.0, 0.0]))
    assert cost == pytest.approx(32.0)
    np.testing.assert_allclose(grad, [8.0, 16.0])

    def residuals_with_factory(_state):
        return jnp.asarray([99.0], dtype=jnp.float64)

    residuals_with_factory._state_objective_value_and_cotangent_from_packed = (
        lambda packed_state, layout: (jnp.asarray(5.0), jnp.ones_like(packed_state) * 0.5)
    )
    opt._residuals_fn = residuals_with_factory
    cost, grad = opt.objective_and_gradient_fun(np.asarray([0.0, 0.0]))
    assert cost == pytest.approx(5.0)
    np.testing.assert_allclose(grad, [0.5, 0.5])
    assert opt._profile["gradient_initial_vjp_cache_hit"]["count"] == 1


def test_residual_linear_operator_products_use_tape_and_residual_transposes(monkeypatch) -> None:
    pytest.importorskip("scipy.sparse.linalg")
    import jax.numpy as jnp
    import vmec_jax.discrete_adjoint as adjoint_module
    import vmec_jax.init_guess as init_guess_module

    opt = _bare_optimizer_for_state_ops()
    opt._static = SimpleNamespace(cfg=SimpleNamespace(lasym=True))
    opt._boundary_from_params = lambda params: params
    opt._solve_exact_with_tape = lambda params, *, return_payload: (
        _state_from_coeffs(r=params[0], rs=params[1]),
        {"tape": "tape", "axis_override": {}},
    )
    opt._lasym_replay_column_chunk = lambda n_params: 4 if n_params >= 2 else None

    def fake_initial_guess(_static, boundary, _indata, *, vmec_project, axis_override):
        del vmec_project, axis_override
        return _state_from_coeffs(r=boundary[0], rs=boundary[1])

    def residuals_fn(state_arg):
        r = state_arg.Rcos[0, 0]
        rs = state_arg.Rsin[0, 0]
        return jnp.asarray([r + rs, 2.0 * r - rs], dtype=jnp.float64)

    monkeypatch.setattr(init_guess_module, "initial_guess_from_boundary", fake_initial_guess)
    monkeypatch.setattr(adjoint_module, "checkpoint_tape_state_jvp", lambda **kwargs: kwargs["initial_tangent"])
    monkeypatch.setattr(adjoint_module, "checkpoint_tape_state_vjp", lambda **kwargs: kwargs["final_cotangent"])
    monkeypatch.setattr(adjoint_module, "checkpoint_tape_state_jvp_columns", lambda **kwargs: kwargs["initial_tangents"])

    opt._residuals_fn = residuals_fn
    op = opt.residual_linear_operator(np.asarray([1.0, 2.0]))

    np.testing.assert_allclose(op.matvec(np.asarray([3.0, 4.0])), [7.0, 2.0])
    np.testing.assert_allclose(op.matmat(np.asarray([[1.0, 2.0], [3.0, 4.0]])), [[4.0, 6.0], [-1.0, 0.0]])
    np.testing.assert_allclose(op.rmatvec(np.asarray([5.0, 7.0])), [19.0, -2.0])
    assert op.shape == (2, 2)
    assert opt._profile["linear_operator_matvec"]["count"] == 1
    assert opt._profile["linear_operator_matmat"]["count"] == 1
    assert opt._profile["linear_operator_rmatvec"]["count"] == 1
    assert opt._profile["linear_operator_initial_transpose"]["count"] == 1

    def residual_cotangent_from_packed(_packed_state, _layout, cotangent):
        return jnp.asarray([10.0 * cotangent[0], 20.0 * cotangent[1], 0.0, 0.0, 0.0, 0.0], dtype=jnp.float64)

    residuals_fn._state_cotangent_from_packed = residual_cotangent_from_packed
    op_with_helper = opt.residual_linear_operator(np.asarray([1.0, 2.0]))
    np.testing.assert_allclose(op_with_helper.rmatvec(np.asarray([2.0, 3.0])), [20.0, 60.0])
    assert opt._profile["linear_operator_initial_transpose"]["count"] == 2


def test_qh_and_qs_residual_cotangent_operator_factories(monkeypatch) -> None:
    import jax.numpy as jnp
    import vmec_jax.modes as modes_module
    import vmec_jax.quasisymmetry as qs_module
    import vmec_jax.wout as wout_module
    from vmec_jax.optimization import make_qh_residuals_fn, make_qs_residuals_fn

    static = SimpleNamespace(
        s=np.asarray([0.0, 1.0]),
        cfg=SimpleNamespace(mpol=2, ntor=1, ntheta=3, nzeta=3, nfp=1),
    )
    state = _state_from_coeffs(r=2.0, rs=0.3, z=0.5)
    packed = pack_state(state)

    monkeypatch.setattr(opt_module, "flux_profiles_from_indata", lambda *_args, **_kwargs: "flux")
    monkeypatch.setattr(opt_module, "_pressure_profile_for_static", lambda *_args, **_kwargs: jnp.asarray([0.0, 1.0]))
    monkeypatch.setattr(
        modes_module,
        "nyquist_mode_table_from_grid",
        lambda **_kwargs: SimpleNamespace(m=np.asarray([0]), n=np.asarray([0])),
    )
    monkeypatch.setattr(qs_module, "_quasisymmetry_angle_cache", lambda **_kwargs: "angles")
    monkeypatch.setattr(
        qs_module,
        "quasisymmetry_ratio_residual_from_state",
        lambda **kwargs: {"residuals1d": jnp.asarray([3.0 * kwargs["state"].Zcos[0, 0]]), "total": 0.0},
    )
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

    qh = make_qh_residuals_fn(static, object(), signgs=1, target_aspect=2.5, aspect_weight=2.0, qs_weight=4.0)
    qh_cotangent = qh._state_cotangent_from_packed(packed, state.layout, jnp.asarray([1.0, 1.0]))
    assert np.asarray(qh_cotangent).shape == np.asarray(packed).shape
    assert np.all(np.isfinite(np.asarray(qh_cotangent)))

    qs = make_qs_residuals_fn(
        static,
        object(),
        signgs=1,
        target_aspect=2.5,
        target_iota=0.4,
        aspect_weight=2.0,
        iota_weight=5.0,
        qs_weight=4.0,
    )
    qs_cotangent = qs._state_cotangent_from_packed(packed, state.layout, jnp.asarray([1.0, 1.0, 1.0]))
    assert np.asarray(qs_cotangent).shape == np.asarray(packed).shape
    assert np.all(np.isfinite(np.asarray(qs_cotangent)))


def test_exact_residual_after_jacobian_evaluates_cached_state_on_cache_miss() -> None:
    opt = _bare_optimizer_for_state_ops()
    opt._last_jacobian_key = [b"accepted"]
    opt._last_jacobian_residual = None
    opt._exact_cache = {b"accepted": ("state", {"payload": True})}
    opt._cached_exact_residual = lambda *args, **kwargs: None
    opt._evaluate_residuals_from_state = lambda state: np.asarray([7.0, 8.0]) if state == "state" else np.asarray([])

    np.testing.assert_allclose(opt._exact_residual_after_jacobian(), [7.0, 8.0])
    np.testing.assert_allclose(opt._exact_residual_cache[b"accepted"], [7.0, 8.0])


def test_post_jacobian_clear_invokes_global_cache_clearers(monkeypatch) -> None:
    import vmec_jax.discrete_adjoint as adjoint_module
    import vmec_jax.preconditioner_1d_jax as precond_module
    import vmec_jax.vmec_numpy_forces as numpy_forces_module

    calls = []
    monkeypatch.setattr(adjoint_module, "clear_replay_scan_caches", lambda: calls.append("replay"))
    monkeypatch.setattr(precond_module, "clear_preconditioner_jit_caches", lambda: calls.append("precond"))
    monkeypatch.setattr(numpy_forces_module, "clear_numpy_force_caches", lambda: calls.append("numpy"))

    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._post_jacobian_clear(clear_compiled=False)
    assert calls == []
    opt._post_jacobian_clear(clear_compiled=True)
    assert calls == ["replay", "precond", "numpy"]


def test_gauss_newton_ftol_termination_after_accepted_step() -> None:
    result = opt_module.gauss_newton_least_squares(
        lambda x: np.asarray([float(x[0]) - 1.0]),
        lambda _x: np.asarray([[1.0]]),
        np.asarray([0.0]),
        max_nfev=5,
        ftol=2.0,
        gtol=0.0,
        xtol=0.0,
        verbose=0,
    )

    assert result["success"] is True
    assert result["message"] == "`ftol` termination condition is satisfied."


def _run_ready_optimizer_for_method_tests(residual: np.ndarray | None = None) -> FixedBoundaryExactOptimizer:
    residual = np.asarray([1.0, 2.0] if residual is None else residual, dtype=float)
    state = SimpleNamespace(name="state")
    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._scan_exact_path = "tape"
    opt._solver_device_name = None
    opt._inside_solver_device_context = False
    opt._trial_residual_cache = OrderedDict()
    opt._exact_cache = {}
    opt._exact_state_cache = {}
    opt._exact_residual_cache = {}
    opt._initial_tangent_cache = {}
    opt._profile = {}
    opt._static = SimpleNamespace()
    opt._inner_max_iter = 1
    opt._inner_ftol = 1.0e-9
    opt._trial_max_iter = 1
    opt._trial_ftol = 1.0e-6
    opt._post_jacobian_clear = lambda *args, **kwargs: None
    opt._profile_dump = lambda: {}
    opt._exact_cache_key = lambda params: np.asarray(params, dtype=float).round(12).reshape(-1).tobytes()
    opt._base_params_vector = lambda: np.zeros(1)
    opt._cached_exact_residual = lambda *args, **kwargs: residual.copy()
    opt._cached_exact_state = lambda _params: state
    opt._evaluate_residuals_from_state = lambda _state: residual.copy()
    opt._solve_exact_with_tape = lambda _params, return_payload=False: (state, {}) if return_payload else state
    opt._solve_forward = lambda _params, trial=True: state
    opt._qs_total_from_state = lambda _state, res: float(np.dot(np.asarray(res, dtype=float), np.asarray(res, dtype=float)))
    opt.residual_fun = lambda _params: residual.copy()
    opt.forward_residual_fun = lambda _params: residual.copy()
    opt._jacobian_fun_tracked = lambda _params: np.asarray([[1.0], [0.0]], dtype=float)
    opt._remember_best_exact_point = FixedBoundaryExactOptimizer._remember_best_exact_point.__get__(
        opt, FixedBoundaryExactOptimizer
    )
    return opt


def _install_exact_point_fallback_fixture(
    opt: FixedBoundaryExactOptimizer,
    residual_by_point: dict[float, np.ndarray],
) -> None:
    states: dict[float, SimpleNamespace] = {}

    opt._cached_exact_residual = FixedBoundaryExactOptimizer._cached_exact_residual.__get__(
        opt, FixedBoundaryExactOptimizer
    )
    opt._cached_exact_state = FixedBoundaryExactOptimizer._cached_exact_state.__get__(opt, FixedBoundaryExactOptimizer)

    def remember_point(params):
        point = float(np.asarray(params, dtype=float).reshape(-1)[0])
        if point not in residual_by_point:
            raise RuntimeError(f"no exact solve for {point}")
        state = states.setdefault(point, SimpleNamespace(name=f"state-{point:g}"))
        key = opt._exact_cache_key(np.asarray([point], dtype=float))
        opt._exact_cache[key] = (state, {})
        opt._remember_exact_state(key, state)
        opt._remember_exact_residual(key, residual_by_point[point])
        return state

    def solve_exact(params, return_payload=False):
        state = remember_point(params)
        return (state, {}) if return_payload else state

    opt._solve_exact_with_tape = solve_exact
    opt.residual_fun = lambda params: np.asarray(
        residual_by_point[float(np.asarray(params, dtype=float).reshape(-1)[0])],
        dtype=float,
    )
    opt.forward_residual_fun = lambda _params: np.asarray([99.0], dtype=float)
    opt._remember_exact_test_point = remember_point


def test_run_scipy_exception_returns_best_exact_and_records_iota_targets(monkeypatch) -> None:
    pytest.importorskip("scipy.optimize")
    import scipy.optimize

    monkeypatch.setattr("vmec_jax.wout.equilibrium_aspect_ratio_from_state", lambda **_kwargs: 7.0)
    monkeypatch.setattr(scipy.optimize, "least_squares", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    opt = _run_ready_optimizer_for_method_tests(np.asarray([0.5, 0.25]))
    result = opt.run(
        np.asarray([0.0]),
        method="scipy",
        max_nfev=3,
        verbose=0,
        iota_fn=lambda _state: 0.42,
        target_iota=0.4,
        target_aspect=7.0,
    )

    assert result["success"] is False
    assert result["status"] == -1
    assert "returning best exact accepted point" in result["message"]
    assert result["_history_dump"]["selected_best_exact_point"] is True
    assert result["_history_dump"]["target_iota"] == pytest.approx(0.4)
    assert result["_history_dump"]["target_aspect"] == pytest.approx(7.0)
    assert result["_history_dump"]["iota_final"] == pytest.approx(0.42)
    assert "optimizer_exception" in result["_history_dump"]


def test_run_scipy_final_exact_failure_uses_prior_best_exact_not_trial(monkeypatch) -> None:
    pytest.importorskip("scipy.optimize")
    import scipy.optimize

    monkeypatch.setattr("vmec_jax.wout.equilibrium_aspect_ratio_from_state", lambda **_kwargs: 7.0)

    def fake_least_squares(*args, **kwargs):
        return SimpleNamespace(
            x=np.asarray([1.0]),
            cost=0.0,
            nfev=2,
            njev=1,
            success=True,
            status=1,
            message="nominal scipy success",
        )

    monkeypatch.setattr(scipy.optimize, "least_squares", fake_least_squares)
    opt = _run_ready_optimizer_for_method_tests(np.asarray([0.5, 0.25]))
    opt._cached_exact_residual = lambda *args, **kwargs: None
    opt._cached_exact_state = lambda params: "best-exact-state" if np.allclose(np.asarray(params), [0.0]) else None

    def solve_exact(params, return_payload=False):
        if np.allclose(np.asarray(params), [0.0]):
            return ("best-exact-state", {}) if return_payload else "best-exact-state"
        raise RuntimeError("final exact solve failed")

    opt._solve_exact_with_tape = solve_exact
    opt._solve_forward = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("trial fallback was used"))

    result = opt.run(np.asarray([0.0]), method="scipy", max_nfev=3, verbose=0)

    np.testing.assert_allclose(result["x"], [0.0])
    assert result["_state_final"] == "best-exact-state"
    assert result["_history_dump"]["selected_best_exact_point"] is True


def test_run_scalar_trust_history_remembers_accepted_exact_point(monkeypatch) -> None:
    monkeypatch.setattr("vmec_jax.wout.equilibrium_aspect_ratio_from_state", lambda **_kwargs: 7.0)

    opt = _run_ready_optimizer_for_method_tests(np.asarray([2.0]))
    _install_exact_point_fallback_fixture(
        opt,
        {
            0.0: np.asarray([2.0]),
            1.0: np.asarray([0.25]),
        },
    )

    def objective_and_gradient(params):
        point = float(np.asarray(params, dtype=float).reshape(-1)[0])
        opt._remember_exact_test_point(np.asarray([point]))
        residual = np.asarray({0.0: [2.0], 1.0: [0.25]}[point], dtype=float)
        gradient = np.asarray([-1.0 if point == 0.0 else 0.0], dtype=float)
        return 0.5 * float(np.dot(residual, residual)), gradient

    opt.objective_and_gradient_fun = objective_and_gradient

    result = opt.run(np.asarray([0.0]), method="scalar_trust", max_nfev=2, scalar_step_bound=1.0, verbose=0)

    np.testing.assert_allclose(result["x"], [1.0])
    np.testing.assert_allclose(opt._best_exact_params, [1.0])
    np.testing.assert_allclose(opt._best_exact_residual, [0.25])


def test_run_lbfgs_history_best_exact_drives_final_fallback(monkeypatch) -> None:
    pytest.importorskip("scipy.optimize")
    import scipy.optimize

    monkeypatch.setattr("vmec_jax.wout.equilibrium_aspect_ratio_from_state", lambda **_kwargs: 7.0)

    def fake_minimize(fun, y0, *, jac, method, bounds, options):
        assert jac is True
        fun(np.asarray(y0, dtype=float))
        fun(np.asarray([1.0], dtype=float))
        return SimpleNamespace(
            x=np.asarray([2.0]),
            fun=0.0,
            success=True,
            status=0,
            message="lbfgs returned unreplayable final point",
            nit=1,
        )

    monkeypatch.setattr(scipy.optimize, "minimize", fake_minimize)
    opt = _run_ready_optimizer_for_method_tests(np.asarray([2.0]))
    _install_exact_point_fallback_fixture(
        opt,
        {
            0.0: np.asarray([2.0]),
            1.0: np.asarray([0.25]),
        },
    )

    def objective_and_gradient(params):
        point = float(np.asarray(params, dtype=float).reshape(-1)[0])
        opt._remember_exact_test_point(np.asarray([point]))
        residual = np.asarray({0.0: [2.0], 1.0: [0.25]}[point], dtype=float)
        gradient = np.asarray([-1.0 if point == 0.0 else 0.0], dtype=float)
        return 0.5 * float(np.dot(residual, residual)), gradient

    opt.objective_and_gradient_fun = objective_and_gradient

    result = opt.run(np.asarray([0.0]), method="lbfgs_adjoint", max_nfev=3, verbose=0)

    np.testing.assert_allclose(result["x"], [1.0])
    assert result["_state_final"].name == "state-1"
    assert result["_history_dump"]["selected_best_exact_point"] is True


def test_run_lbfgs_success_path_uses_scaled_bounds_and_result(monkeypatch) -> None:
    pytest.importorskip("scipy.optimize")
    import scipy.optimize

    monkeypatch.setattr("vmec_jax.wout.equilibrium_aspect_ratio_from_state", lambda **_kwargs: 7.0)
    calls = {}

    def fake_minimize(fun, y0, *, jac, method, bounds, options):
        calls.update({"y0": np.asarray(y0), "jac": jac, "method": method, "bounds": bounds, "options": options})
        cost, grad = fun(np.asarray(y0, dtype=float))
        assert np.isfinite(cost)
        np.testing.assert_allclose(grad, [-2.0])
        return SimpleNamespace(
            x=np.asarray([0.75]),
            fun=0.125,
            success=True,
            status=0,
            message="lbfgs ok",
            nit=2,
        )

    monkeypatch.setattr(scipy.optimize, "minimize", fake_minimize)
    opt = _run_ready_optimizer_for_method_tests(np.asarray([0.5]))
    opt.objective_and_gradient_fun = lambda x: (
        0.5 * float((np.asarray(x)[0] - 1.0) ** 2),
        np.asarray([np.asarray(x)[0] - 1.0], dtype=float),
    )

    result = opt.run(
        np.asarray([0.0]),
        method="lbfgs_adjoint",
        max_nfev=4,
        x_scale=np.asarray([2.0]),
        lbfgs_step_bound=0.25,
        verbose=0,
    )

    np.testing.assert_allclose(calls["y0"], [0.0])
    assert calls["method"] == "L-BFGS-B"
    assert calls["bounds"] == [(-0.25, 0.25)]
    assert calls["options"]["maxfun"] == 4
    np.testing.assert_allclose(result["x"], [1.5])
    assert result["success"] is True
    assert result["message"] == "lbfgs ok"
    assert result["_history_dump"]["method"] == "lbfgs_adjoint"


def test_run_matrix_free_residual_callback_uses_state_cache_then_trial(monkeypatch) -> None:
    pytest.importorskip("scipy.optimize")
    import scipy.optimize

    monkeypatch.setattr("vmec_jax.wout.equilibrium_aspect_ratio_from_state", lambda **_kwargs: 7.0)
    residual_calls = []

    class FakeOperator:
        shape = (2, 1)

        def matvec(self, vector):
            return np.asarray([vector[0], 2.0 * vector[0]], dtype=float)

        def matmat(self, matrix):
            return np.vstack([matrix[0], 2.0 * matrix[0]])

        def rmatvec(self, vector):
            return np.asarray([np.sum(vector)], dtype=float)

    def fake_least_squares(residuals, y0, *, jac, **kwargs):
        residual_calls.append(np.asarray(residuals(y0), dtype=float).copy())
        residual_calls.append(np.asarray(residuals(y0 + 1.0), dtype=float).copy())
        op = jac(y0)
        np.testing.assert_allclose(op.matvec(np.asarray([2.0])), [4.0, 8.0])
        assert kwargs["tr_options"] is None
        return SimpleNamespace(
            x=np.asarray([0.0]),
            cost=0.5,
            nfev=2,
            njev=None,
            success=True,
            status=1,
            message="matrix free ok",
        )

    monkeypatch.setattr(scipy.optimize, "least_squares", fake_least_squares)
    opt = _run_ready_optimizer_for_method_tests(np.asarray([0.25]))
    opt._cached_exact_residual = lambda *args, **kwargs: None
    opt._cached_exact_state = lambda x: "cached-state" if np.allclose(np.asarray(x), [0.0]) else None
    opt._evaluate_residuals_from_state = (
        lambda state: np.asarray([1.0, 2.0]) if state == "cached-state" else np.asarray([0.25])
    )
    opt.forward_residual_fun = lambda _x: np.asarray([9.0, 10.0])
    opt.residual_linear_operator = lambda _x: FakeOperator()

    result = opt.run(np.asarray([0.0]), method="scipy_matrix_free", max_nfev=3, x_scale=np.asarray([2.0]), verbose=0)

    np.testing.assert_allclose(residual_calls[0], [1.0, 2.0])
    np.testing.assert_allclose(residual_calls[1], [9.0, 10.0])
    assert result["njev"] == 0
    assert result["_history_dump"]["method"] == "scipy_matrix_free"


def test_run_matrix_free_history_best_exact_drives_final_fallback(monkeypatch) -> None:
    pytest.importorskip("scipy.optimize")
    import scipy.optimize

    monkeypatch.setattr("vmec_jax.wout.equilibrium_aspect_ratio_from_state", lambda **_kwargs: 7.0)

    class FakeOperator:
        shape = (1, 1)

        def matvec(self, vector):
            return np.asarray(vector, dtype=float)

        def matmat(self, matrix):
            return np.asarray(matrix, dtype=float)

        def rmatvec(self, vector):
            return np.asarray(vector, dtype=float)

    def fake_least_squares(residuals, y0, *, jac, **kwargs):
        np.testing.assert_allclose(residuals(y0), [2.0])
        jac(np.asarray([1.0], dtype=float))
        return SimpleNamespace(
            x=np.asarray([2.0]),
            cost=0.0,
            nfev=2,
            njev=1,
            success=True,
            status=1,
            message="matrix-free returned unreplayable final point",
        )

    monkeypatch.setattr(scipy.optimize, "least_squares", fake_least_squares)
    opt = _run_ready_optimizer_for_method_tests(np.asarray([2.0]))
    _install_exact_point_fallback_fixture(
        opt,
        {
            0.0: np.asarray([2.0]),
            1.0: np.asarray([0.25]),
        },
    )
    opt.residual_linear_operator = lambda params: opt._remember_exact_test_point(params) and FakeOperator()

    result = opt.run(np.asarray([0.0]), method="scipy_matrix_free", max_nfev=3, verbose=0)

    np.testing.assert_allclose(result["x"], [1.0])
    assert result["_state_final"].name == "state-1"
    assert result["_history_dump"]["selected_best_exact_point"] is True


def test_run_final_history_wall_time_is_monotone_after_expensive_initial_solve(monkeypatch) -> None:
    pytest.importorskip("scipy.optimize")
    import scipy.optimize

    monkeypatch.setattr("vmec_jax.wout.equilibrium_aspect_ratio_from_state", lambda **_kwargs: 7.0)

    opt = _run_ready_optimizer_for_method_tests(np.asarray([0.25]))

    def fake_least_squares(residuals, y0, *, jac, **kwargs):
        np.testing.assert_allclose(residuals(y0), [0.25])
        opt._history.append(
            {
                "wall_time_s": 42.0,
                "cost": 0.125,
                "objective": 0.25,
                "qs_objective": 0.25,
                "aspect": 7.0,
            }
        )
        return SimpleNamespace(
            x=np.asarray([0.0]),
            cost=0.03125,
            nfev=1,
            njev=0,
            success=True,
            status=1,
            message="ok",
        )

    monkeypatch.setattr(scipy.optimize, "least_squares", fake_least_squares)

    result = opt.run(np.asarray([0.0]), method="scipy", max_nfev=1, verbose=0)
    history = result["_history_dump"]["history"]
    wall_times = [float(entry["wall_time_s"]) for entry in history]

    assert wall_times == sorted(wall_times)
    assert result["_history_dump"]["total_wall_time_s"] == pytest.approx(wall_times[-1])
    assert wall_times[-1] >= 42.0


def test_save_wrappers_write_wout_input_and_history(monkeypatch, tmp_path) -> None:
    opt = _run_ready_optimizer_for_method_tests()
    opt._static = SimpleNamespace(cfg="cfg")
    opt._indata = "indata"
    opt._flux = "flux"
    opt._signgs = -1
    opt._profile_add = FixedBoundaryExactOptimizer._profile_add.__get__(opt, FixedBoundaryExactOptimizer)
    opt._profile = {}
    opt._indata_from_params = lambda params: {"params": np.asarray(params, dtype=float).tolist()}
    opt._cached_exact_state = lambda _params: None
    solve_calls = []
    opt._solve_forward = lambda _params, trial=True: solve_calls.append(bool(trial)) or "exact-state"

    captured = {}
    monkeypatch.setattr(
        "vmec_jax.driver.write_wout_from_fixed_boundary_run",
        lambda path, run, **kwargs: captured.update({"wout_path": path, "run": run, "wout_kwargs": kwargs}),
    )
    monkeypatch.setattr(opt_module, "write_indata", lambda path, indata: captured.update({"input": (path, indata)}))

    opt.save_wout(tmp_path / "nested" / "wout.nc", params=np.asarray([1.0]))
    opt.save_input(tmp_path / "nested" / "input.test", np.asarray([2.0]))
    opt.save_history(tmp_path / "nested" / "history.json", {"_history_dump": {"ok": True}})

    assert solve_calls == [False]
    assert captured["run"].state == "exact-state"
    assert captured["wout_kwargs"] == {"include_fsq": False, "fast_bcovar": True}
    assert captured["input"][1] == {"params": [2.0]}
    assert (tmp_path / "nested" / "history.json").read_text().strip() == '{\n  "ok": true\n}'
    with pytest.raises(ValueError, match="requires either params or state"):
        opt.save_wout(tmp_path / "bad.nc")
