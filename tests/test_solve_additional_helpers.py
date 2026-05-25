from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax._compat import jnp
import vmec_jax.solve as solve_module
from vmec_jax.solve import (
    _can_reassemble_precond_mats,
    _ForceBlocks,
    _apply_vmec_lambda_axis_rules_to_state,
    _finite_float_or_zero,
    _format_axis_coeff,
    _format_checkpoint_log_row,
    _format_evolve_trace_row,
    _format_freeb_control_trace_row,
    _format_time_control_log_row,
    _format_time_control_trace_row,
    _format_vmec2000_iter_row,
    _free_boundary_iter_controls,
    _grad_rms_state,
    _half_mesh_from_full_mesh,
    _host_restart_decision,
    _initial_axis_reset_decision,
    _jit_cache_get,
    _jit_cache_limit,
    _jit_cache_put,
    _lambda_preconditioned_full_norm,
    _mask_grad_for_constraints,
    _materialize_adjoint_trace_array,
    _merge_axis_reset_state,
    _metric_surface_precond_scales_jax,
    _metric_surface_precond_scales_np,
    _mode_weight_force_blocks_np,
    _mask_scan_restart_force_payload,
    _normalize_adjoint_trace_mode,
    _normalize_resume_state_mode,
    _pack_resume_state_record,
    _preconditioner_output_blocks_np,
    _pshalf_from_s_jax,
    _pshalf_from_s_np,
    _radial_tridi_smooth_dirichlet,
    _replace_mode_slice,
    _replace_mode_slice_np,
    _append_residual_iter_history_record,
    _append_residual_iter_terminal_history,
    _residual_iter_history_record,
    _safe_dt_from_force_blocks,
    _scale_m1_precond_rhs_from_mats,
    _resolve_cg_tol,
    _resolve_grad_tol,
    _resolve_lbfgs_curvature_tol,
    _resolve_lm_damping,
    _s_half_from_full_mesh_s,
    _scale_mode_slice,
    _scale_mode_slice_np,
    _scale_velocity_blocks,
    _sm_sp_from_s_np,
    _update_state_gd,
    _should_print_vmec2000_row,
    _vmec2000_time_control_decision,
    _vmec2000_scan_options_from_env,
    _vmec_force_flux_profiles,
    _vmec_scale_m1_factors_from_mats,
    _write_axis_reset_dump,
    _vmec2000_cadence_selected,
    _free_boundary_prev_rz_fsq_next,
    _free_boundary_should_damp_constraint_baseline,
    _free_boundary_turnon_resets_iter1_immediately,
    _zero_coeff_column,
    _zero_coeff_column_np,
    _zero_edge_rz_force_block,
    _zero_edge_rz_force_blocks,
    _zero_velocity_blocks_like,
    first_step_diagnostics,
    solve_lambda_gd,
    solve_fixed_boundary_gn_vmec_residual,
    solve_fixed_boundary_lbfgs_vmec_residual,
)
from vmec_jax.vmec_tomnsp import TomnspsRZL
from vmec_jax.state import StateLayout, VMECState


def _state_from_value(value: float, *, ns: int = 3, k: int = 3) -> VMECState:
    layout = StateLayout(ns=ns, K=k, lasym=False)
    arr = np.full((ns, k), float(value), dtype=float)
    return VMECState(
        layout=layout,
        Rcos=arr.copy(),
        Rsin=arr.copy(),
        Zcos=arr.copy(),
        Zsin=arr.copy(),
        Lcos=arr.copy(),
        Lsin=arr.copy(),
    )


def _lambda_solver_static(*, ns: int = 2, k: int = 2):
    return SimpleNamespace(
        cfg=SimpleNamespace(nfp=1),
        modes=SimpleNamespace(m=np.arange(k, dtype=int), n=np.zeros(k, dtype=int)),
        basis=object(),
        s=np.linspace(0.0, 1.0, ns),
        grid=SimpleNamespace(theta=np.asarray([0.0, np.pi]), zeta=np.asarray([0.0])),
    )


def _scan_options(**overrides):
    kwargs = dict(
        verbose=True,
        vmec2000_control=True,
        verbose_vmec2000_table=True,
        light_history=False,
        scan_minimal_default=None,
        dump_any=False,
        fsq_total_target=None,
        backend_name="cpu",
        force_chunked_scan_run=False,
        scan_print_env="1",
        scan_print_mode_env="debug_callback",
        scan_print_ordered_env="0",
        scan_print_chunked_env="1",
        scan_light_env="0",
        scan_minimal_env="",
        scan_core_env="",
        scan_trace_env="0",
        abort_scan_env="0",
        scan_precompute_env="",
        tridi_precompute_env="",
        scan_lax_env="",
        tridi_solve_env="",
        scan_restart_payload_env="",
    )
    kwargs.update(overrides)
    return _vmec2000_scan_options_from_env(**kwargs)


def test_vmec2000_scan_options_quiet_runs_default_to_minimal_history():
    opts = _scan_options(verbose=False)

    assert opts.scan_minimal
    assert not opts.scan_light
    assert not opts.scan_collect_scalars
    assert not opts.scan_collect_print
    assert not opts.print_in_scan
    assert not opts.chunked_print
    assert opts.scan_use_restart_payload


def test_vmec2000_scan_options_dump_forces_full_history_and_print_chunking():
    opts = _scan_options(
        light_history=True,
        scan_light_env="yes",
        scan_minimal_env="1",
        dump_any=True,
        scan_print_ordered_env="true",
        scan_print_mode_env="bogus",
        scan_core_env="minimal",
    )

    assert not opts.scan_minimal
    assert not opts.scan_light
    assert opts.scan_collect_scalars
    assert opts.scan_collect_print
    assert opts.scan_print_ordered
    assert opts.scan_print_mode == "debug_print"
    assert opts.chunked_print
    assert not opts.print_in_scan


def test_vmec2000_scan_options_env_overrides_preconditioner_and_restart_flags():
    default_cpu_lax = _scan_options(backend_name="cpu", scan_lax_env="", tridi_solve_env="")
    default_gpu_lax = _scan_options(backend_name="gpu", scan_lax_env="", tridi_solve_env="")
    assert not default_cpu_lax.scan_use_lax_tridi
    assert not default_gpu_lax.scan_use_lax_tridi
    assert default_cpu_lax.scan_use_precomputed
    assert default_gpu_lax.scan_use_precomputed

    opts = _scan_options(
        backend_name="gpu",
        scan_precompute_env="0",
        tridi_precompute_env="1",
        scan_lax_env="yes",
        tridi_solve_env="",
        scan_restart_payload_env="",
        scan_trace_env="1",
        abort_scan_env="true",
        force_chunked_scan_run=True,
    )

    assert not opts.scan_use_precomputed
    assert opts.scan_use_lax_tridi
    assert not opts.scan_use_restart_payload
    assert opts.scan_trace
    assert opts.abort_scan_on_badjac
    assert opts.chunked_print
    assert not opts.print_in_scan


def test_vmec2000_scan_options_restart_payload_explicit_env_wins():
    disabled = _scan_options(backend_name="cpu", scan_restart_payload_env="false")
    enabled = _scan_options(backend_name="gpu", scan_restart_payload_env="YES")

    assert not disabled.scan_use_restart_payload
    assert enabled.scan_use_restart_payload


def test_scan_restart_force_payload_zeros_residual_impulse_and_invalidates_cache():
    force_blocks = tuple(
        jnp.asarray(np.full((2, 2), float(idx + 1))) for idx in range(12)
    )

    masked_blocks, cache_valid = _mask_scan_restart_force_payload(
        force_blocks=force_blocks,
        cache_valid=jnp.asarray(True),
        do_restart=jnp.asarray(True),
    )

    for block in masked_blocks:
        np.testing.assert_allclose(np.asarray(block), 0.0)
    assert not bool(np.asarray(cache_valid))
    assert sum(float(np.asarray(jnp.sum(block * block))) for block in masked_blocks) == 0.0


def test_scan_restart_force_payload_preserves_current_residuals_without_restart():
    force_blocks = tuple(
        jnp.asarray(np.arange(4, dtype=float).reshape(2, 2) + 10.0 * idx)
        for idx in range(12)
    )

    masked_blocks, cache_valid = _mask_scan_restart_force_payload(
        force_blocks=force_blocks,
        cache_valid=jnp.asarray(True),
        do_restart=jnp.asarray(False),
    )

    for actual, expected in zip(masked_blocks, force_blocks):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected))
    assert bool(np.asarray(cache_valid))
    np.testing.assert_allclose(
        sum(float(np.asarray(jnp.sum(block * block))) for block in masked_blocks),
        sum(float(np.asarray(jnp.sum(block * block))) for block in force_blocks),
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"w_rz": -1.0}, "w_rz and w_l"),
        ({"w_l": -1.0}, "w_rz and w_l"),
        ({"objective_scale": 0.0}, "objective_scale"),
        ({"scale_rz": 0.0}, "scale_rz and scale_l"),
        ({"scale_l": 0.0}, "scale_rz and scale_l"),
        ({"history_size": 0}, "history_size"),
        ({"max_iter": 0}, "max_iter"),
        ({"max_backtracks": -1}, "max_backtracks"),
        ({"bt_factor": 1.0}, "bt_factor"),
    ],
)
def test_lbfgs_vmec_residual_rejects_invalid_solver_controls(kwargs, message):
    state = _state_from_value(0.0)
    with pytest.raises(ValueError, match=message):
        solve_fixed_boundary_lbfgs_vmec_residual(
            state,
            SimpleNamespace(),
            indata=SimpleNamespace(),
            signgs=1,
            verbose=False,
            **kwargs,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"damping_increase": 1.0}, "damping_increase"),
        ({"damping_decrease": 0.0}, "damping_decrease"),
        ({"damping_decrease": 1.1}, "damping_decrease"),
        ({"max_damping": 0.0}, "max_damping"),
        ({"max_retries": -1}, "max_retries"),
        ({"zero_m1_iters": -1}, "zero_m1_iters"),
        ({"zero_m1_fsqz_thresh": -1.0}, "zero_m1_fsqz_thresh"),
        ({"w_rz": -1.0}, "w_rz and w_l"),
        ({"w_l": -1.0}, "w_rz and w_l"),
        ({"max_iter": 0}, "max_iter"),
        ({"cg_maxiter": 0}, "cg_maxiter"),
        ({"bt_factor": 1.0}, "bt_factor"),
        ({"objective_scale": 0.0}, "objective_scale"),
    ],
)
def test_gn_vmec_residual_rejects_invalid_solver_controls(kwargs, message):
    state = _state_from_value(0.0)
    with pytest.raises(ValueError, match=message):
        solve_fixed_boundary_gn_vmec_residual(
            state,
            SimpleNamespace(),
            indata=SimpleNamespace(),
            signgs=1,
            verbose=False,
            **kwargs,
        )


def _lambda_solver_state(*, ns: int = 2, k: int = 2) -> VMECState:
    layout = StateLayout(ns=ns, K=k, lasym=False)
    zeros = np.zeros((ns, k), dtype=float)
    lcos = np.zeros((ns, k), dtype=float)
    lsin = np.zeros((ns, k), dtype=float)
    lcos[:, 1] = np.asarray([0.3, -0.2])[:ns]
    lsin[:, 1] = np.asarray([-0.4, 0.25])[:ns]
    return VMECState(
        layout=layout,
        Rcos=np.ones((ns, k), dtype=float),
        Rsin=zeros.copy(),
        Zcos=zeros.copy(),
        Zsin=np.ones((ns, k), dtype=float),
        Lcos=lcos,
        Lsin=lsin,
    )


def _patch_tiny_lambda_problem(monkeypatch):
    pytest.importorskip("jax")

    from vmec_jax._compat import jnp

    def fake_eval_geom(state, _static):
        shape = jnp.asarray(state.Lcos).shape
        return SimpleNamespace(
            g_tt=jnp.ones(shape),
            g_tp=jnp.zeros(shape),
            g_pp=2.0 * jnp.ones(shape),
            sqrtg=jnp.ones(shape),
        )

    monkeypatch.setattr(solve_module, "eval_geom", fake_eval_geom)
    monkeypatch.setattr(
        solve_module,
        "eval_fourier_dtheta",
        lambda Lcos, _Lsin, *_args, **_kwargs: jnp.asarray(Lcos),
    )
    monkeypatch.setattr(
        solve_module,
        "eval_fourier_dzeta_phys",
        lambda _Lcos, Lsin, *_args, **_kwargs: jnp.asarray(Lsin),
    )
    monkeypatch.setattr(
        solve_module,
        "bsup_from_sqrtg_lambda",
        lambda *, lam_u, lam_v, phipf, chipf, lamscale, **_kwargs: (
            lam_u + 0.1 * jnp.asarray(phipf)[:, None],
            lam_v + 0.2 * jnp.asarray(chipf)[:, None] * jnp.asarray(lamscale),
        ),
    )


def test_solve_lambda_gd_tiny_problem_descends_with_mode_preconditioner(monkeypatch, capsys):
    _patch_tiny_lambda_problem(monkeypatch)
    state = _lambda_solver_state()
    static = _lambda_solver_static()

    res = solve_lambda_gd(
        state,
        static,
        phipf=np.ones(2),
        chipf=np.asarray([0.3, 0.4]),
        signgs=1,
        lamscale=np.asarray([1.0, 1.1]),
        max_iter=5,
        step_size=0.2,
        grad_tol=0.0,
        max_backtracks=3,
        jit_grad=True,
        preconditioner="mode_diag",
        precond_exponent=1.0,
        verbose=True,
    )

    assert res.n_iter >= 1
    assert np.isfinite(res.wb_history).all()
    assert np.all(np.diff(res.wb_history) < 0.0)
    assert float(res.wb_history[-1]) < float(res.wb_history[0])
    assert res.diagnostics["idx00"] == 0
    assert res.diagnostics["grad_tol"] == 0.0
    np.testing.assert_allclose(np.asarray(res.state.Lcos)[:, 0], 0.0)
    np.testing.assert_allclose(np.asarray(res.state.Lsin)[:, 0], 0.0)
    assert "[solve_lambda_gd] iter=000" in capsys.readouterr().out


def test_solve_lambda_gd_line_search_failure_stops_without_accepting(monkeypatch, capsys):
    _patch_tiny_lambda_problem(monkeypatch)
    state = _lambda_solver_state()

    res = solve_lambda_gd(
        state,
        _lambda_solver_static(),
        phipf=np.ones(2),
        chipf=np.ones(2),
        signgs=1,
        lamscale=np.ones(2),
        max_iter=3,
        step_size=0.0,
        grad_tol=0.0,
        max_backtracks=2,
        verbose=True,
    )

    assert res.n_iter == 0
    assert res.wb_history.shape == (1,)
    np.testing.assert_allclose(np.asarray(res.state.Lcos), np.asarray(state.Lcos))
    np.testing.assert_allclose(np.asarray(res.state.Lsin), np.asarray(state.Lsin))
    np.testing.assert_allclose(res.step_history, [0.0])
    assert "line search failed" in capsys.readouterr().out


def test_solve_lambda_gd_validates_solver_controls(monkeypatch):
    _patch_tiny_lambda_problem(monkeypatch)
    state = _lambda_solver_state()
    static = _lambda_solver_static()
    common = dict(phipf=np.ones(2), chipf=np.ones(2), signgs=1, lamscale=np.ones(2), verbose=False)

    with pytest.raises(ValueError, match="max_iter"):
        solve_lambda_gd(state, static, max_iter=0, **common)
    with pytest.raises(ValueError, match="max_backtracks"):
        solve_lambda_gd(state, static, max_backtracks=-1, **common)
    with pytest.raises(ValueError, match="bt_factor"):
        solve_lambda_gd(state, static, bt_factor=1.0, **common)
    with pytest.raises(ValueError, match="Unknown preconditioner"):
        solve_lambda_gd(state, static, preconditioner="bad", **common)
    with pytest.raises(ValueError, match="precond_exponent"):
        solve_lambda_gd(state, static, preconditioner="mode_diag", precond_exponent=0.0, **common)


def test_jit_cache_limit_put_and_lru_policy(monkeypatch):
    monkeypatch.setenv("VMEC_JAX_TEST_CACHE", "-4")
    assert _jit_cache_limit("VMEC_JAX_TEST_CACHE", 3) == 0

    monkeypatch.setenv("VMEC_JAX_TEST_CACHE", "not-an-int")
    assert _jit_cache_limit("VMEC_JAX_TEST_CACHE", 3) == 3

    cache: OrderedDict[tuple, object] = OrderedDict()
    monkeypatch.setenv("VMEC_JAX_TEST_CACHE", "0")
    value = object()
    assert _jit_cache_put(cache, ("disabled",), value, env_name="VMEC_JAX_TEST_CACHE", default=2) is value
    assert cache == {}

    monkeypatch.setenv("VMEC_JAX_TEST_CACHE", "2")
    _jit_cache_put(cache, ("a",), "A", env_name="VMEC_JAX_TEST_CACHE", default=2)
    _jit_cache_put(cache, ("b",), "B", env_name="VMEC_JAX_TEST_CACHE", default=2)
    assert _jit_cache_get(cache, ("a",)) == "A"
    _jit_cache_put(cache, ("c",), "C", env_name="VMEC_JAX_TEST_CACHE", default=2)
    assert list(cache.keys()) == [("a",), ("c",)]
    assert _jit_cache_get(cache, ("missing",)) is None


def test_mode_slice_helpers_cover_invalid_none_and_singleton_branches():
    arr = np.arange(6, dtype=float).reshape(2, 3)
    np.testing.assert_allclose(np.asarray(_zero_coeff_column(arr, idx=-1)), arr)
    np.testing.assert_allclose(np.asarray(_zero_coeff_column(arr, idx=3)), arr)
    np.testing.assert_allclose(np.asarray(_zero_coeff_column(np.ones((2, 1)), idx=0)), np.zeros((2, 1)))
    np.testing.assert_allclose(_zero_coeff_column_np(arr, idx=1), np.array([[0.0, 0.0, 2.0], [3.0, 0.0, 5.0]]))

    cube = np.arange(2 * 3 * 2, dtype=float).reshape(2, 3, 2)
    repl = np.full((2, 2), -5.0)
    assert _replace_mode_slice(None, mode_idx=0, replacement=repl) is None
    assert _scale_mode_slice(None, mode_idx=0, scale=np.ones(2)) is None
    np.testing.assert_allclose(np.asarray(_replace_mode_slice(cube, mode_idx=9, replacement=repl)), cube)
    np.testing.assert_allclose(np.asarray(_scale_mode_slice(cube, mode_idx=-1, scale=np.ones(2))), cube)

    one_mode = cube[:, :1, :]
    np.testing.assert_allclose(np.asarray(_replace_mode_slice(one_mode, mode_idx=0, replacement=repl)), repl[:, None, :])
    np.testing.assert_allclose(_replace_mode_slice_np(one_mode, mode_idx=0, replacement=repl), repl[:, None, :])
    np.testing.assert_allclose(_scale_mode_slice_np(one_mode, mode_idx=0, scale=np.array([2.0, 3.0])), one_mode * np.array([2.0, 3.0])[:, None, None])


def test_state_update_mask_and_rms_helpers_are_componentwise():
    pytest.importorskip("jax")

    state = _state_from_value(2.0)
    grad = _state_from_value(1.0)
    updated = _update_state_gd(state, grad, step=0.25, scale_rz=2.0, scale_l=4.0)
    for field in ("Rcos", "Rsin", "Zcos", "Zsin"):
        np.testing.assert_allclose(np.asarray(getattr(updated, field)), 1.5)
    for field in ("Lcos", "Lsin"):
        np.testing.assert_allclose(np.asarray(getattr(updated, field)), 1.0)

    assert _grad_rms_state(grad) == pytest.approx(np.sqrt(6.0))

    static = SimpleNamespace(modes=SimpleNamespace(m=np.array([0, 1, 2])))
    masked = _mask_grad_for_constraints(grad, static, idx00=0, mask_lambda_axis=False)
    for field in ("Rcos", "Rsin", "Zcos", "Zsin"):
        got = np.asarray(getattr(masked, field))
        np.testing.assert_allclose(got[-1, :], 0.0)
        np.testing.assert_allclose(got[0, :], np.array([1.0, 0.0, 0.0]))
    for field in ("Lcos", "Lsin"):
        got = np.asarray(getattr(masked, field))
        np.testing.assert_allclose(got[:, 0], 0.0)
        np.testing.assert_allclose(got[0, 1:], 1.0)


def test_tolerance_resolvers_validate_explicit_values_and_scale_by_dtype():
    assert _resolve_grad_tol(0.0, grad_rms0=10.0, dtype=np.float64) == 0.0
    with pytest.raises(ValueError, match="grad_tol"):
        _resolve_grad_tol(-1.0, grad_rms0=10.0, dtype=np.float64)
    assert _resolve_grad_tol(None, grad_rms0=4.0, dtype=np.float32) == pytest.approx(
        np.sqrt(np.finfo(np.float32).eps) * 4.0
    )

    with pytest.raises(ValueError, match="cg_tol"):
        _resolve_cg_tol(0.0, current_obj=1.0, initial_obj=1.0, target_obj=0.0, dtype=np.float64)
    assert _resolve_cg_tol(None, current_obj=1.0, initial_obj=3.0, target_obj=0.0, dtype=np.float64) == pytest.approx(0.25)

    with pytest.raises(ValueError, match="damping"):
        _resolve_lm_damping(-1.0, curvature_scale=2.0, dtype=np.float64)
    assert _resolve_lm_damping(None, curvature_scale=2.0, dtype=np.float64) == pytest.approx(
        np.sqrt(np.finfo(np.float64).eps) * 2.0
    )

    assert _resolve_lbfgs_curvature_tol(np.array([3.0, 4.0]), np.array([0.0, 6.0])) == pytest.approx(
        np.finfo(float).eps * 30.0
    )


def test_mesh_flux_and_free_boundary_cadence_helpers():
    np.testing.assert_allclose(np.asarray(_s_half_from_full_mesh_s(np.array([0.0]))), np.array([0.0]))
    np.testing.assert_allclose(
        np.asarray(_s_half_from_full_mesh_s(np.array([0.0, 0.25, 1.0]))),
        np.array([0.0, 0.125, 0.625]),
    )
    np.testing.assert_allclose(
        np.asarray(_half_mesh_from_full_mesh(np.array([2.0, 4.0, 10.0]))),
        np.array([2.0, 3.0, 7.0]),
    )

    phipf_internal, chipf_internal, chips_eff = _vmec_force_flux_profiles(
        phipf=np.array([2.0, 4.0]),
        chipf=None,
        signgs=1,
        flux_is_internal=True,
    )
    np.testing.assert_allclose(np.asarray(phipf_internal), np.array([2.0, 4.0]))
    assert chipf_internal is None
    np.testing.assert_allclose(np.asarray(chips_eff), np.zeros(2))

    phipf_external, _, chips_iota = _vmec_force_flux_profiles(
        phipf=np.array([2.0 * np.pi, 4.0 * np.pi]),
        chipf=None,
        signgs=1,
        flux_is_internal=False,
        iotaf=np.array([3.0, 5.0]),
    )
    np.testing.assert_allclose(np.asarray(phipf_external), np.array([1.0, 2.0]))
    np.testing.assert_allclose(np.asarray(chips_iota), np.array([3.0, 10.0]))

    assert _free_boundary_iter_controls(iter2=5, iter1=1, nvacskip=0) == (1, 0)
    assert _free_boundary_iter_controls(iter2=6, iter1=1, nvacskip=4) == (2, 1)


def test_resume_state_mode_and_payload_packing(monkeypatch):
    monkeypatch.setenv("VMEC_JAX_RESUME_STATE_MODE", "light")
    assert _normalize_resume_state_mode(None) == "minimal"
    assert _normalize_resume_state_mode(" compact ") == "minimal"
    assert _normalize_resume_state_mode("off") == "none"
    assert _normalize_resume_state_mode("") == "full"
    with pytest.raises(ValueError, match="resume_state_mode"):
        _normalize_resume_state_mode("huge")

    base = {"time_step": 0.1, "iter1": 3}
    heavy = {"cache": object(), "iter1": 9}
    assert _pack_resume_state_record(base=base, heavy=heavy, mode="minimal") == base
    assert _pack_resume_state_record(base=base, heavy=heavy, mode="none") is None

    full = _pack_resume_state_record(base=base, heavy=heavy, mode="full")
    assert full is not None
    assert full["time_step"] == 0.1
    assert full["iter1"] == 9
    assert "cache" in full
    assert base["iter1"] == 3


def test_vmec2000_cadence_and_row_formatting_helpers():
    assert _vmec2000_cadence_selected(iter_idx=1, max_iter=20, nstep_screen=7)
    assert _vmec2000_cadence_selected(iter_idx=20, max_iter=20, nstep_screen=7)
    assert _vmec2000_cadence_selected(iter_idx=14, max_iter=20, nstep_screen=7)
    assert not _vmec2000_cadence_selected(iter_idx=13, max_iter=20, nstep_screen=7)
    assert not _should_print_vmec2000_row(
        iter_idx=1,
        max_iter=20,
        nstep_screen=7,
        verbose=False,
        vmec2000_control=True,
        verbose_vmec2000_table=True,
    )

    row = _format_vmec2000_iter_row(
        iter_idx=12,
        fsqr=1.25,
        fsqz=2.5,
        fsql=3.75,
        delt0r=0.125,
        r00=4.5,
        w_mhd=6.25,
        lasym=False,
    )
    assert row == "   12  1.25E+00  2.50E+00  3.75E+00  4.500E+00  1.25E-01  6.2500E+00"

    row_lasym = _format_vmec2000_iter_row(
        iter_idx=2,
        fsqr=1.0,
        fsqz=2.0,
        fsql=3.0,
        delt0r=4.0,
        r00=5.0,
        z00=None,
        w_mhd=6.0,
        lasym=True,
    )
    assert "NAN" in row_lasym
    assert row_lasym.startswith("    2  1.00E+00")


def test_trace_formatting_and_scalar_guard_helpers():
    assert _format_axis_coeff(1.0e-5) == "1E-05"
    assert _finite_float_or_zero(3.5) == 3.5
    assert _finite_float_or_zero(np.asarray(np.nan)) == 0.0
    assert _finite_float_or_zero(np.asarray(np.inf)) == 0.0

    assert _format_time_control_log_row(
        iter_idx=4,
        fsq=1.0,
        fsq0=2.0,
        res0=3.0,
        res1=4.0,
        time_step=0.5,
    ) == "iter=4 fsq=1.000000e+00 fsq0=2.000000e+00 res0=3.000000e+00 res1=4.000000e+00 time_step=5.000000e-01\n"

    trace_row = _format_time_control_trace_row(
        stage="restart",
        iter2=9,
        iter1=3,
        fsq=1.0,
        fsq0=2.0,
        res0=3.0,
        res1=4.0,
        time_step=0.5,
        irst=2,
    )
    assert trace_row.endswith("   2 restart\n")
    assert " 1.0000000000000000e+00" in trace_row

    assert _format_checkpoint_log_row(iter_idx=8, fsq=1.0, fsq0=2.0, res0=3.0, res1=4.0).startswith("iter=8")
    assert _format_freeb_control_trace_row(
        iter2=2,
        iter1=1,
        ivac=3,
        ivacskip=0,
        nvacskip=5,
        fsq_rz_prev=0.25,
        cached=True,
    ).endswith(" 1\n")
    assert _format_evolve_trace_row(
        iter2=2,
        iter1=1,
        ns=3,
        stage="pre",
        fsq1=1.0,
        fsq_prev=2.0,
        time_step=0.5,
        dtau=0.25,
        b1=0.75,
        fac=0.8,
        xc_norm=10.0,
        v_norm=11.0,
        g_norm=12.0,
    ).startswith("       2        1        3 pre")


def test_legacy_dump_guard_helpers_and_time_control_append(monkeypatch, tmp_path):
    monkeypatch.delenv("VMEC_JAX_DUMP_TIMECONTROL", raising=False)
    monkeypatch.delenv("VMEC_JAX_DUMP_DIR", raising=False)
    assert solve_module._legacy_dump_record_path(
        enable_env="VMEC_JAX_DUMP_TIMECONTROL",
        filename="time_control.log",
    ) is None

    monkeypatch.setenv("VMEC_JAX_DUMP_TIMECONTROL", "0")
    monkeypatch.setenv("VMEC_JAX_DUMP_DIR", str(tmp_path))
    assert solve_module._legacy_dump_record_path(
        enable_env="VMEC_JAX_DUMP_TIMECONTROL",
        filename="time_control.log",
    ) is None

    monkeypatch.setenv("VMEC_JAX_DUMP_TIMECONTROL", "false")
    assert solve_module._legacy_dump_record_path(
        enable_env="VMEC_JAX_DUMP_TIMECONTROL",
        filename="time_control.log",
    ) == tmp_path / "time_control.log"

    solve_module._maybe_dump_time_control_record(
        iter_idx=4,
        fsq=1.0,
        fsq0=2.0,
        res0=3.0,
        res1=4.0,
        time_step=0.5,
    )
    assert (tmp_path / "time_control.log").read_text(encoding="utf-8") == (
        "iter=4 fsq=1.000000e+00 fsq0=2.000000e+00 "
        "res0=3.000000e+00 res1=4.000000e+00 time_step=5.000000e-01\n"
    )


def test_legacy_single_dump_iter_filter_matches_invalid_as_all():
    assert solve_module._legacy_single_dump_iter_selected(dump_iter="", iter_idx=3)
    assert solve_module._legacy_single_dump_iter_selected(dump_iter="3", iter_idx=3)
    assert not solve_module._legacy_single_dump_iter_selected(dump_iter="2", iter_idx=3)
    assert solve_module._legacy_single_dump_iter_selected(dump_iter="not-an-int", iter_idx=3)


def test_radial_tridi_smoothing_matches_dense_dirichlet_reference():
    pytest.importorskip("jax")

    rhs = np.array(
        [
            [1.0, 2.0],
            [4.0, 8.0],
            [9.0, 18.0],
            [16.0, 32.0],
        ]
    )
    alpha = 0.25
    system = np.array(
        [
            [1.0 + 2.0 * alpha, -alpha],
            [-alpha, 1.0 + 2.0 * alpha],
        ]
    )
    interior_rhs = rhs[1:-1].copy()
    interior_rhs[0] += alpha * rhs[0]
    interior_rhs[-1] += alpha * rhs[-1]
    expected = rhs.copy()
    expected[1:-1] = np.linalg.solve(system, interior_rhs)

    smoothed = _radial_tridi_smooth_dirichlet(rhs, alpha=alpha)
    np.testing.assert_allclose(np.asarray(smoothed), expected)

    rhs3 = rhs.reshape(4, 2, 1)
    smoothed3 = _radial_tridi_smooth_dirichlet(rhs3, alpha=alpha)
    np.testing.assert_allclose(np.asarray(smoothed3), expected.reshape(4, 2, 1))

    assert _radial_tridi_smooth_dirichlet(rhs, alpha=0.0, skip_nonpositive=True) is rhs
    with pytest.raises(ValueError, match="ndim>=2"):
        _radial_tridi_smooth_dirichlet(np.arange(3.0), alpha=alpha)
    with pytest.raises(ValueError, match="expected \\(ns,K\\) or \\(ns,M,N\\)"):
        _radial_tridi_smooth_dirichlet(np.zeros((3, 1, 1, 1)), alpha=alpha)


def test_first_step_metric_mesh_and_adjoint_trace_helper_branches():
    pytest.importorskip("jax")

    guu = np.array([[[4.0], [1.0]], [[0.0], [0.0]], [[1.0e-12], [1.0e-12]]])
    r12 = np.ones_like(guu)
    bsubu = np.array([[[3.0], [4.0]], [[0.0], [0.0]], [[1.0e12], [1.0e12]]])
    bsubv = np.zeros_like(bsubu)
    w_ang = np.ones((2, 1))

    rz_np, l_np = _metric_surface_precond_scales_np(guu=guu, r12=r12, bsubu=bsubu, bsubv=bsubv, w_ang=w_ang)
    np.testing.assert_allclose(rz_np, np.array([1.0 / np.sqrt(5.0), 1.0, 100.0]))
    np.testing.assert_allclose(l_np, np.array([0.2, 1.0, 1.0e-4]))

    rz_jax, l_jax = _metric_surface_precond_scales_jax(guu=guu, r12=r12, bsubu=bsubu, bsubv=bsubv, w_ang=w_ang)
    np.testing.assert_allclose(np.asarray(rz_jax), rz_np)
    np.testing.assert_allclose(np.asarray(l_jax), l_np)

    s = np.array([0.0, 0.25, 1.0])
    np.testing.assert_allclose(_pshalf_from_s_np(s), np.sqrt(np.array([0.125, 0.125, 0.625])))
    np.testing.assert_allclose(np.asarray(_pshalf_from_s_jax(s, np.float64)), _pshalf_from_s_np(s))
    sm, sp = _sm_sp_from_s_np(s)
    assert sm.shape == (4,)
    assert sp.shape == (4,)
    np.testing.assert_allclose(_sm_sp_from_s_np(np.array([0.0]))[0], np.zeros(2))

    arr = np.array([1.0, 2.0])
    assert _normalize_adjoint_trace_mode(" dynamic ") == "dynamic"
    assert _materialize_adjoint_trace_array(arr, mode="dynamic") is arr
    np.testing.assert_allclose(_materialize_adjoint_trace_array([1.0, 2.0], mode="full"), arr)
    with pytest.raises(ValueError, match="adjoint_trace_mode"):
        _normalize_adjoint_trace_mode("summary")


def test_axis_reset_state_merge_replaces_only_m0_geometry_and_preserves_lambda():
    state = _state_from_value(1.0, ns=2, k=3)
    axis_state = _state_from_value(9.0, ns=2, k=3)
    state = VMECState(
        layout=state.layout,
        Rcos=state.Rcos,
        Rsin=state.Rsin + 1.0,
        Zcos=state.Zcos + 2.0,
        Zsin=state.Zsin + 3.0,
        Lcos=state.Lcos + 4.0,
        Lsin=state.Lsin + 5.0,
    )
    axis_state = VMECState(
        layout=axis_state.layout,
        Rcos=axis_state.Rcos,
        Rsin=axis_state.Rsin + 1.0,
        Zcos=axis_state.Zcos + 2.0,
        Zsin=axis_state.Zsin + 3.0,
        Lcos=axis_state.Lcos + 4.0,
        Lsin=axis_state.Lsin + 5.0,
    )
    static = SimpleNamespace(modes=SimpleNamespace(m=np.array([0, 1, 0])))

    merged = _merge_axis_reset_state(st=state, st_axis=axis_state, static=static, full_reset=False)
    np.testing.assert_allclose(np.asarray(merged.Rcos), np.array([[9.0, 1.0, 9.0], [9.0, 1.0, 9.0]]))
    np.testing.assert_allclose(np.asarray(merged.Rsin), np.array([[10.0, 2.0, 10.0], [10.0, 2.0, 10.0]]))
    np.testing.assert_allclose(np.asarray(merged.Zcos), np.array([[11.0, 3.0, 11.0], [11.0, 3.0, 11.0]]))
    np.testing.assert_allclose(np.asarray(merged.Zsin), np.array([[12.0, 4.0, 12.0], [12.0, 4.0, 12.0]]))
    np.testing.assert_allclose(np.asarray(merged.Lcos), np.asarray(state.Lcos))
    np.testing.assert_allclose(np.asarray(merged.Lsin), np.asarray(state.Lsin))

    full = _merge_axis_reset_state(st=state, st_axis=axis_state, static=static, full_reset=True)
    assert full is axis_state


def test_axis_reset_state_merge_uses_cached_m0_mask_when_available():
    state = _state_from_value(2.0, ns=2, k=3)
    axis_state = _state_from_value(8.0, ns=2, k=3)
    static = SimpleNamespace(
        modes=SimpleNamespace(m=np.array([99, 99, 99])),
        m_is_m0=np.asarray([0.0, 1.0, 0.0]),
    )

    merged = _merge_axis_reset_state(st=state, st_axis=axis_state, static=static, full_reset=False)

    np.testing.assert_allclose(np.asarray(merged.Rcos), np.array([[2.0, 8.0, 2.0], [2.0, 8.0, 2.0]]))
    np.testing.assert_allclose(np.asarray(merged.Zsin), np.array([[2.0, 8.0, 2.0], [2.0, 8.0, 2.0]]))
    np.testing.assert_allclose(np.asarray(merged.Lcos), np.asarray(state.Lcos))


def test_initial_axis_reset_decision_requires_state_confirmation_when_configured():
    decision = _initial_axis_reset_decision(
        bad_jacobian_ptau=True,
        bad_jacobian_state=False,
        badjac_use_state=True,
        fsq_phys=2.0,
        axis_reset_fsq_min=1.0,
        force_axis_reset=False,
        axis_reset_always_3d=False,
        lthreed=True,
    )

    assert not decision.bad_jacobian
    assert not decision.force_reset
    assert not decision.reset

    confirmed = _initial_axis_reset_decision(
        bad_jacobian_ptau=True,
        bad_jacobian_state=True,
        badjac_use_state=True,
        fsq_phys=2.0,
        axis_reset_fsq_min=1.0,
        force_axis_reset=False,
        axis_reset_always_3d=False,
        lthreed=True,
    )

    assert confirmed.bad_jacobian
    assert confirmed.reset


@pytest.mark.parametrize("fsq_phys", [None, np.nan, 0.5])
def test_initial_axis_reset_decision_residual_floor_suppresses_bad_jacobian(fsq_phys):
    decision = _initial_axis_reset_decision(
        bad_jacobian_ptau=True,
        bad_jacobian_state=False,
        badjac_use_state=False,
        fsq_phys=fsq_phys,
        axis_reset_fsq_min=1.0,
        force_axis_reset=False,
        axis_reset_always_3d=False,
        lthreed=True,
    )

    assert not decision.bad_jacobian
    assert not decision.reset


def test_initial_axis_reset_decision_force_reset_bypasses_residual_floor():
    explicit = _initial_axis_reset_decision(
        bad_jacobian_ptau=False,
        bad_jacobian_state=False,
        badjac_use_state=False,
        fsq_phys=0.0,
        axis_reset_fsq_min=1.0,
        force_axis_reset=True,
        axis_reset_always_3d=False,
        lthreed=False,
        vmec2000_control=False,
        lmove_axis=False,
    )
    three_d = _initial_axis_reset_decision(
        bad_jacobian_ptau=None,
        bad_jacobian_state=False,
        badjac_use_state=False,
        fsq_phys=None,
        axis_reset_fsq_min=1.0,
        force_axis_reset=False,
        axis_reset_always_3d=True,
        lthreed=True,
        vmec2000_control=True,
        lmove_axis=True,
    )
    disabled = _initial_axis_reset_decision(
        bad_jacobian_ptau=True,
        bad_jacobian_state=False,
        badjac_use_state=False,
        fsq_phys=2.0,
        axis_reset_fsq_min=1.0,
        force_axis_reset=False,
        axis_reset_always_3d=False,
        lthreed=True,
        axis_reset_enabled=False,
    )

    assert explicit.force_reset
    assert explicit.reset
    assert three_d.force_reset
    assert three_d.reset
    assert disabled.bad_jacobian
    assert not disabled.reset


def test_write_axis_reset_dump_validates_optional_diagnostic_file(tmp_path):
    coeffs = np.asarray([1.0, 2.0, 3.0])

    assert not _write_axis_reset_dump(
        axis_dump_dir="",
        ns=5,
        ntor=2,
        used_state_guess=False,
        raxis_cc=coeffs,
        raxis_cs=coeffs,
        zaxis_cc=coeffs,
        zaxis_cs=coeffs,
    )

    assert _write_axis_reset_dump(
        axis_dump_dir=tmp_path,
        ns=5,
        ntor=2,
        used_state_guess=True,
        raxis_cc=coeffs,
        raxis_cs=coeffs + 10.0,
        zaxis_cc=coeffs + 20.0,
        zaxis_cs=coeffs + 30.0,
    )

    text = (tmp_path / "axis_reset_ns5.dat").read_text(encoding="utf-8")
    assert "# used_state_guess=1" in text
    assert "n raxis_cc raxis_cs zaxis_cc zaxis_cs" in text
    assert "   2  3.0000000000000000e+00  1.3000000000000000e+01" in text


def test_write_axis_reset_dump_returns_false_for_incomplete_coefficients(tmp_path):
    assert not _write_axis_reset_dump(
        axis_dump_dir=tmp_path,
        ns=3,
        ntor=2,
        used_state_guess=False,
        raxis_cc=np.asarray([1.0]),
        raxis_cs=np.asarray([0.0]),
        zaxis_cc=np.asarray([0.0]),
        zaxis_cs=np.asarray([0.0]),
    )
    assert not (tmp_path / "axis_reset_ns3.dat").exists()


def test_zero_edge_rz_force_blocks_preserves_lambda_and_short_mesh_numpy_identity():
    one_row = np.asarray([[1.0, 2.0]])
    assert _zero_edge_rz_force_block(one_row) is one_row

    block = np.arange(6.0).reshape(3, 2)
    zeroed = _zero_edge_rz_force_block(block)
    assert zeroed is not block
    np.testing.assert_allclose(zeroed[:-1], block[:-1])
    np.testing.assert_allclose(zeroed[-1], 0.0)
    np.testing.assert_allclose(block[-1], [4.0, 5.0])

    frzl = TomnspsRZL(
        frcc=block,
        frss=block + 10.0,
        fzsc=block + 20.0,
        fzcs=block + 30.0,
        flsc=block + 40.0,
        flcs=block + 50.0,
        frsc=block + 60.0,
        frcs=block + 70.0,
        fzcc=block + 80.0,
        fzss=block + 90.0,
        flcc=block + 100.0,
        flss=block + 110.0,
    )

    masked = _zero_edge_rz_force_blocks(frzl)

    for name in ("frcc", "frss", "fzsc", "fzcs", "frsc", "frcs", "fzcc", "fzss"):
        np.testing.assert_allclose(np.asarray(getattr(masked, name))[-1], 0.0)
    for name in ("flsc", "flcs", "flcc", "flss"):
        np.testing.assert_allclose(np.asarray(getattr(masked, name)), np.asarray(getattr(frzl, name)))


def test_free_boundary_turnon_cadence_branches_follow_vmec_controls():
    assert _free_boundary_prev_rz_fsq_next(
        prev_fsq_before=7.0,
        fsq_rz_curr=0.25,
        turnon_restart=True,
        preserve_turnon_restart=True,
    ) == pytest.approx(7.0)
    assert _free_boundary_prev_rz_fsq_next(
        prev_fsq_before=7.0,
        fsq_rz_curr=0.25,
        turnon_restart=True,
        preserve_turnon_restart=False,
    ) == pytest.approx(0.25)

    assert _free_boundary_should_damp_constraint_baseline(freeb_ivac=1, freeb_turnon_iter=False, lthreed=True)
    assert not _free_boundary_should_damp_constraint_baseline(freeb_ivac=1, freeb_turnon_iter=True, lthreed=True)
    assert _free_boundary_should_damp_constraint_baseline(freeb_ivac=1, freeb_turnon_iter=True, lthreed=False)
    assert not _free_boundary_should_damp_constraint_baseline(freeb_ivac=-1, freeb_turnon_iter=False, lthreed=False)

    assert _free_boundary_turnon_resets_iter1_immediately(lthreed=False, lasym=True)
    assert _free_boundary_turnon_resets_iter1_immediately(lthreed=True, lasym=False)
    assert not _free_boundary_turnon_resets_iter1_immediately(lthreed=True, lasym=True)


def test_vmec_scale_m1_factors_jax_and_reassembly_contract():
    pytest.importorskip("jax")

    parity_mats = {
        "ard_parity": np.array([[1.0, 2.0], [1.0, 0.0]]),
        "brd_parity": np.array([[1.0, 4.0], [1.0, 0.0]]),
        "azd_parity": np.array([[1.0, 6.0], [1.0, 0.0]]),
        "bzd_parity": np.array([[1.0, 8.0], [1.0, 0.0]]),
    }
    fac_r, fac_z = _vmec_scale_m1_factors_from_mats(parity_mats)
    np.testing.assert_allclose(np.asarray(fac_r), np.array([6.0 / 20.0, 1.0]))
    np.testing.assert_allclose(np.asarray(fac_z), np.array([14.0 / 20.0, 1.0]))

    fallback_mats = {
        "dr": -np.array([[[0.0], [3.0]], [[0.0], [0.0]]]),
        "dz": -np.array([[[0.0], [9.0]], [[0.0], [0.0]]]),
    }
    fac_r, fac_z = _vmec_scale_m1_factors_from_mats(fallback_mats)
    np.testing.assert_allclose(np.asarray(fac_r), np.array([0.25, 1.0]))
    np.testing.assert_allclose(np.asarray(fac_z), np.array([0.75, 1.0]))

    assert not _can_reassemble_precond_mats(None)
    complete = {key: object() for key in (
        "arm_parity",
        "ard_parity",
        "brm_parity",
        "brd_parity",
        "azm_parity",
        "azd_parity",
        "bzm_parity",
        "bzd_parity",
        "cxd_full",
        "delta_s",
    )}
    assert _can_reassemble_precond_mats(complete)


def test_safe_dt_from_force_blocks_limits_finite_forces_and_preserves_bad_rms_nominal():
    force = np.full((2, 2), 3.0)
    blocks = _ForceBlocks(
        frcc=force,
        frss=None,
        fzsc=np.full((2, 2), 4.0),
        fzcs=None,
        flsc=np.zeros((2, 2)),
        flcs=None,
        frsc=None,
        frcs=None,
        fzcc=None,
        fzss=None,
        flcc=None,
        flss=None,
    )

    dt = _safe_dt_from_force_blocks(dt_nominal=0.1, max_coeff_delta_rms=1.0e-4, blocks=blocks)

    assert dt == pytest.approx(np.sqrt(1.0e-4 / 5.0))
    zero_dt = _safe_dt_from_force_blocks(
        dt_nominal=0.25,
        max_coeff_delta_rms=1.0e-4,
        blocks=blocks._replace(frcc=np.zeros((2, 2)), fzsc=np.zeros((2, 2))),
    )
    assert zero_dt == pytest.approx(0.25)
    bad_dt = _safe_dt_from_force_blocks(
        dt_nominal=0.25,
        max_coeff_delta_rms=1.0e-4,
        blocks=blocks._replace(frcc=np.full((2, 2), np.inf)),
    )
    assert bad_dt == pytest.approx(0.25)


def test_apply_vmec_lambda_axis_rules_zeroes_only_gauge_column_and_can_be_disabled():
    state = _state_from_value(1.0, ns=2, k=3)
    state = VMECState(
        layout=state.layout,
        Rcos=state.Rcos,
        Rsin=state.Rsin,
        Zcos=state.Zcos,
        Zsin=state.Zsin,
        Lcos=np.arange(6.0).reshape(2, 3) + 10.0,
        Lsin=np.arange(6.0).reshape(2, 3) + 20.0,
    )

    assert (
        _apply_vmec_lambda_axis_rules_to_state(
            state,
            enforce_vmec_lambda_axis=False,
            host_update_assembly=True,
            idx00=1,
        )
        is state
    )

    host = _apply_vmec_lambda_axis_rules_to_state(
        state,
        enforce_vmec_lambda_axis=True,
        host_update_assembly=True,
        idx00=1,
    )
    np.testing.assert_allclose(np.asarray(host.Lcos)[:, 1], 0.0)
    np.testing.assert_allclose(np.asarray(host.Lsin)[:, 1], 0.0)
    np.testing.assert_allclose(np.asarray(host.Lcos)[:, [0, 2]], np.asarray(state.Lcos)[:, [0, 2]])
    np.testing.assert_allclose(np.asarray(state.Lcos)[:, 1], [11.0, 14.0])

    jax_state = _apply_vmec_lambda_axis_rules_to_state(
        state,
        enforce_vmec_lambda_axis=True,
        host_update_assembly=False,
        idx00=2,
    )
    np.testing.assert_allclose(np.asarray(jax_state.Lcos)[:, 2], 0.0)
    np.testing.assert_allclose(np.asarray(jax_state.Lsin)[:, 2], 0.0)


def test_scale_m1_precond_rhs_from_mats_scales_m1_slice_and_extends_factor_tail():
    shape = (3, 2, 1)
    base = np.arange(np.prod(shape), dtype=float).reshape(shape) + 1.0
    frzl = TomnspsRZL(
        frcc=base,
        frss=base + 10.0,
        fzsc=base + 20.0,
        fzcs=base + 30.0,
        flsc=base + 40.0,
        flcs=base + 50.0,
        frsc=base + 60.0,
        frcs=base + 70.0,
        fzcc=base + 80.0,
        fzss=base + 90.0,
        flcc=base + 100.0,
        flss=base + 110.0,
    )
    mats = {
        "ard_parity": np.array([[0.0, 1.0], [0.0, 3.0]]),
        "brd_parity": np.array([[0.0, 1.0], [0.0, 1.0]]),
        "azd_parity": np.array([[0.0, 3.0], [0.0, 2.0]]),
        "bzd_parity": np.array([[0.0, 3.0], [0.0, 2.0]]),
    }
    fac_r = np.array([0.25, 0.5, 1.0])
    fac_z = np.array([0.75, 0.5, 1.0])

    assert _scale_m1_precond_rhs_from_mats(
        frzl,
        mats,
        lconm1=False,
        mpol=2,
        host_update_assembly=True,
    ) is frzl

    host = _scale_m1_precond_rhs_from_mats(
        frzl,
        mats,
        lconm1=True,
        mpol=2,
        host_update_assembly=True,
    )
    np.testing.assert_allclose(host.frss[:, 1, 0], frzl.frss[:, 1, 0] * fac_r)
    np.testing.assert_allclose(host.fzcs[:, 1, 0], frzl.fzcs[:, 1, 0] * fac_z)
    np.testing.assert_allclose(host.frsc[:, 1, 0], frzl.frsc[:, 1, 0] * fac_r)
    np.testing.assert_allclose(host.fzcc[:, 1, 0], frzl.fzcc[:, 1, 0] * fac_z)
    np.testing.assert_allclose(host.frss[:, 0, 0], frzl.frss[:, 0, 0])
    np.testing.assert_allclose(host.flsc, frzl.flsc)

    pytest.importorskip("jax")
    jax_scaled = _scale_m1_precond_rhs_from_mats(
        frzl,
        mats,
        lconm1=True,
        mpol=2,
        host_update_assembly=False,
    )
    np.testing.assert_allclose(np.asarray(jax_scaled.frss)[:, 1, 0], frzl.frss[:, 1, 0] * fac_r)
    np.testing.assert_allclose(np.asarray(jax_scaled.fzcs)[:, 1, 0], frzl.fzcs[:, 1, 0] * fac_z)


def test_host_preconditioner_output_helpers_preserve_optional_blocks_and_zero_reuse():
    base = np.arange(8.0).reshape(2, 2, 2) + 1.0
    lam = np.linspace(1.0, 2.0, 8).reshape(2, 2, 2)
    frzl_rz = SimpleNamespace(
        frcc=base,
        frss=None,
        fzsc=base + 20.0,
        fzcs=base + 30.0,
        flsc=base + 40.0,
        flcs=None,
        frsc=base + 60.0,
        frcs=None,
        fzcc=base + 80.0,
        fzss=None,
        flcc=base + 100.0,
        flss=None,
    )

    blocks = _preconditioner_output_blocks_np(frzl_rz=frzl_rz, lam_prec=lam)

    assert blocks.frss is None
    assert blocks.flcs is None
    assert blocks.frcs is None
    np.testing.assert_allclose(blocks.frcc, base)
    np.testing.assert_allclose(blocks.flsc, (base + 40.0) * lam)
    np.testing.assert_allclose(blocks.flcc, (base + 100.0) * lam)

    zeros = np.zeros_like(base)
    weight = np.array([[1.0, 2.0], [3.0, 4.0]])
    weighted = _mode_weight_force_blocks_np(blocks, w_mode_mn=weight, zeros_coeff=zeros)

    assert weighted.frss is zeros
    assert weighted.flcs is zeros
    assert weighted.frcs is zeros
    assert weighted.fzss is zeros
    np.testing.assert_allclose(weighted.frcc, base * weight[None, :, :])
    np.testing.assert_allclose(weighted.flsc, (base + 40.0) * lam * weight[None, :, :])
    np.testing.assert_allclose(weighted.frsc, (base + 60.0) * weight[None, :, :])


def test_lambda_preconditioned_full_norm_sums_present_non_axis_rows():
    flsc = np.arange(12.0).reshape(3, 2, 2)
    flcs = np.full_like(flsc, 2.0)
    flcc = np.full_like(flsc, 3.0)
    frzl_pre = SimpleNamespace(flsc=flsc, flcs=flcs, flcc=flcc, flss=None)
    expected = np.sum(flsc[1:] * flsc[1:]) + np.sum(flcs[1:] * flcs[1:]) + np.sum(flcc[1:] * flcc[1:])

    assert _lambda_preconditioned_full_norm(frzl_pre, use_jax=False) == pytest.approx(expected)

    pytest.importorskip("jax")
    assert float(np.asarray(_lambda_preconditioned_full_norm(frzl_pre, use_jax=True))) == pytest.approx(expected)


def test_host_restart_decision_covers_stage_growth_progress_and_vmec_paths():
    base = dict(
        iter2=12,
        iter1=1,
        fsqr=0.0,
        fsqz=0.0,
        fsql=0.0,
        fsq1=1.0,
        fsq_prev=1.0,
        res0=1.0,
        bad_growth_streak=0,
        pre_restart_reason="none",
        reference_mode=False,
        vmec2000_control=False,
        bad_jacobian=False,
        stage_prev_fsq=None,
        stage_transition_factor=50.0,
        lmove_axis=True,
        vmecpp_restart=False,
        k_preconditioner_update_interval=25,
    )

    store_checkpoint = _host_restart_decision(**{**base, "fsq1": 0.5})
    assert store_checkpoint.res0 == pytest.approx(0.5)
    assert store_checkpoint.res0_old == pytest.approx(1.0)
    assert store_checkpoint.store_checkpoint

    bad_growth = _host_restart_decision(**{**base, "fsq1": 101.0, "fsq_prev": 200.0, "bad_growth_streak": 1})
    assert bad_growth.bad_growth_streak == 2
    assert bad_growth.pre_restart_reason == "bad_jacobian"

    bad_progress = _host_restart_decision(**{**base, "iter2": 60, "res0": 10.0, "fsq1": 60.0, "fsq_prev": 61.0})
    assert bad_progress.bad_growth_streak == 0
    assert bad_progress.pre_restart_reason == "bad_progress"

    stage_transition = _host_restart_decision(
        **{**base, "iter2": 1, "iter1": 1, "fsqr": 20.0, "fsqz": 20.0, "fsql": 20.0, "stage_prev_fsq": 1.0}
    )
    assert stage_transition.fsq == pytest.approx(60.0)
    assert stage_transition.pre_restart_reason == "stage_transition"
    assert not stage_transition.huge_initial_forces

    huge_initial = _host_restart_decision(
        **{**base, "iter2": 1, "iter1": 1, "fsqr": np.inf, "stage_prev_fsq": 1.0}
    )
    assert huge_initial.huge_initial_forces

    reference_bad_jac = _host_restart_decision(**{**base, "iter2": 2, "reference_mode": True, "bad_jacobian": True, "fsqr": 11.0})
    assert reference_bad_jac.fsq_res == pytest.approx(11.0)
    assert reference_bad_jac.pre_restart_reason == "bad_jacobian"

    vmec2000_bad_jac = _host_restart_decision(
        **{**base, "iter2": 2, "vmec2000_control": True, "bad_jacobian": True, "fsq1": 0.5, "res0": 2.0}
    )
    assert vmec2000_bad_jac.res0 == pytest.approx(0.5)
    assert vmec2000_bad_jac.pre_restart_reason == "bad_jacobian"

    vmecpp_progress = _host_restart_decision(
        **{**base, "iter2": 60, "fsqr": 0.02, "vmecpp_restart": True}
    )
    assert vmecpp_progress.vmecpp_bad_progress
    assert vmecpp_progress.pre_restart_reason == "bad_progress_vmecpp"


def test_vmec2000_time_control_decision_initializes_and_checkpoints_minima():
    initialized = _vmec2000_time_control_decision(
        iter2=3,
        iter1=3,
        fsq_prev=1.5,
        fsq0_curr=2.5,
        fsq0_prev=9.0,
        res0=-1.0,
        res1=4.0,
        bad_jacobian=True,
        vmec2000_fact=1.0e4,
    )

    assert initialized.fsq == pytest.approx(1.5)
    assert initialized.fsq0 == pytest.approx(2.5)
    assert initialized.res0 == pytest.approx(1.5)
    assert initialized.res1 == pytest.approx(2.5)
    assert initialized.trace_irst == 1
    assert initialized.irst == 1
    assert initialized.initialized
    assert initialized.store_checkpoint
    assert not initialized.restart
    assert initialized.pre_restart_reason == "none"

    improved = _vmec2000_time_control_decision(
        iter2=8,
        iter1=3,
        fsq_prev=0.5,
        fsq0_curr=0.25,
        fsq0_prev=9.0,
        res0=1.0,
        res1=0.3,
        bad_jacobian=False,
        vmec2000_fact=1.0e4,
    )

    assert not improved.initialized
    assert improved.res0 == pytest.approx(0.5)
    assert improved.res1 == pytest.approx(0.25)
    assert improved.store_checkpoint
    assert not improved.restart


def test_vmec2000_time_control_decision_bad_jacobian_uses_previous_physical_residual():
    decision = _vmec2000_time_control_decision(
        iter2=9,
        iter1=3,
        fsq_prev=0.5,
        fsq0_curr=999.0,
        fsq0_prev=0.75,
        res0=1.0,
        res1=2.0,
        bad_jacobian=True,
        vmec2000_fact=1.0e4,
    )

    assert decision.fsq == pytest.approx(0.5)
    assert decision.fsq0 == pytest.approx(0.75)
    assert decision.res0 == pytest.approx(0.5)
    assert decision.res1 == pytest.approx(0.75)
    assert decision.trace_irst == 2
    assert decision.irst == 2
    assert not decision.initialized
    assert not decision.store_checkpoint
    assert decision.restart
    assert decision.pre_restart_reason == "bad_jacobian"


def test_vmec2000_time_control_decision_bad_progress_respects_restart_window():
    in_window = _vmec2000_time_control_decision(
        iter2=18,
        iter1=8,
        fsq_prev=2.0,
        fsq0_curr=3.0,
        fsq0_prev=9.0,
        res0=1.0e-4,
        res1=1.0e-4,
        bad_jacobian=False,
        vmec2000_fact=100.0,
    )
    assert in_window.trace_irst == 1
    assert in_window.irst == 1
    assert not in_window.store_checkpoint
    assert not in_window.restart

    past_window = _vmec2000_time_control_decision(
        iter2=19,
        iter1=8,
        fsq_prev=2.0,
        fsq0_curr=3.0,
        fsq0_prev=9.0,
        res0=1.0e-4,
        res1=1.0e-4,
        bad_jacobian=False,
        vmec2000_fact=100.0,
    )

    assert past_window.res0 == pytest.approx(1.0e-4)
    assert past_window.res1 == pytest.approx(1.0e-4)
    assert past_window.trace_irst == 1
    assert past_window.irst == 3
    assert not past_window.store_checkpoint
    assert past_window.restart
    assert past_window.pre_restart_reason == "time_control"


def test_residual_iter_history_record_packs_restart_row_with_free_boundary_cadence():
    rec = _residual_iter_history_record(
        step=0,
        dt_eff=0,
        update_rms=np.asarray(0.25),
        w_curr=6,
        w_try=np.nan,
        w_try_ratio=np.nan,
        restart_path="pre_restart_trigger",
        step_status="restart_bad_progress",
        restart_reason="bad_progress",
        pre_restart_reason="bad_progress",
        time_step=0.125,
        res0=1.5,
        res1=2.5,
        fsq_prev=3.5,
        bad_growth_streak=4,
        iter1=7,
        iter2=8,
        fsqr=1.0,
        fsqz=2.0,
        fsql=3.0,
        free_boundary_enabled=True,
        freeb_ivac=2,
        freeb_ivacskip=0,
    )

    assert rec.step == pytest.approx(0.0)
    assert rec.dt_eff == pytest.approx(0.0)
    assert rec.update_rms == pytest.approx(0.25)
    assert rec.w_curr == pytest.approx(6.0)
    assert np.isnan(rec.w_try)
    assert np.isnan(rec.w_try_ratio)
    assert rec.restart_path == "pre_restart_trigger"
    assert rec.step_status == "restart_bad_progress"
    assert rec.restart_reason == "bad_progress"
    assert rec.pre_restart_reason == "bad_progress"
    assert rec.time_step == pytest.approx(0.125)
    assert rec.res0 == pytest.approx(1.5)
    assert rec.res1 == pytest.approx(2.5)
    assert rec.fsq_prev == pytest.approx(3.5)
    assert rec.bad_growth_streak == 4
    assert rec.iter1 == 7
    assert rec.iter2 == 8
    assert rec.grad_rms == pytest.approx(np.sqrt(6.0))
    assert rec.freeb_ivac == 2
    assert rec.freeb_ivacskip == 0
    assert rec.freeb_full_update == 1


def test_residual_iter_history_record_clamps_negative_total_and_omits_free_boundary():
    rec = _residual_iter_history_record(
        step=0.5,
        dt_eff=0.25,
        update_rms=0.125,
        w_curr=-1.0,
        w_try=2.0,
        w_try_ratio=3.0,
        restart_path="non_strict",
        step_status="momentum",
        restart_reason="none",
        pre_restart_reason="none",
        time_step=4.0,
        res0=5.0,
        res1=6.0,
        fsq_prev=7.0,
        bad_growth_streak=8,
        iter1=9,
        iter2=10,
        fsqr=-10.0,
        fsqz=1.0,
        fsql=2.0,
        free_boundary_enabled=False,
        freeb_ivac=-1,
        freeb_ivacskip=0,
    )

    assert rec.step == pytest.approx(0.5)
    assert rec.dt_eff == pytest.approx(0.25)
    assert rec.grad_rms == pytest.approx(0.0)
    assert rec.freeb_ivac is None
    assert rec.freeb_ivacskip is None
    assert rec.freeb_full_update is None


def test_append_residual_iter_history_record_keeps_all_channels_aligned():
    rec = _residual_iter_history_record(
        step=0.5,
        dt_eff=0.25,
        update_rms=0.125,
        w_curr=1.0,
        w_try=2.0,
        w_try_ratio=3.0,
        restart_path="vmec2000_time_control",
        step_status="restart_time_control",
        restart_reason="time_control",
        pre_restart_reason="time_control",
        time_step=4.0,
        res0=5.0,
        res1=6.0,
        fsq_prev=7.0,
        bad_growth_streak=8,
        iter1=9,
        iter2=10,
        fsqr=9.0,
        fsqz=16.0,
        fsql=0.0,
        free_boundary_enabled=True,
        freeb_ivac=3,
        freeb_ivacskip=0,
    )
    histories = {
        "step_history": [],
        "dt_eff_history": [],
        "update_rms_history": [],
        "w_curr_history": [],
        "w_try_history": [],
        "w_try_ratio_history": [],
        "restart_path_history": [],
        "step_status_history": [],
        "restart_reason_history": [],
        "pre_restart_reason_history": [],
        "time_step_history": [],
        "res0_history": [],
        "res1_history": [],
        "fsq_prev_history": [],
        "bad_growth_streak_history": [],
        "iter1_history": [],
        "iter2_history": [],
        "grad_rms_history": [],
        "freeb_ivac_history": [],
        "freeb_ivacskip_history": [],
        "freeb_full_update_history": [],
    }

    _append_residual_iter_history_record(rec, free_boundary_enabled=True, **histories)

    assert histories["step_history"] == [pytest.approx(0.5)]
    assert histories["dt_eff_history"] == [pytest.approx(0.25)]
    assert histories["update_rms_history"] == [pytest.approx(0.125)]
    assert histories["w_curr_history"] == [pytest.approx(1.0)]
    assert histories["w_try_history"] == [pytest.approx(2.0)]
    assert histories["w_try_ratio_history"] == [pytest.approx(3.0)]
    assert histories["restart_path_history"] == ["vmec2000_time_control"]
    assert histories["step_status_history"] == ["restart_time_control"]
    assert histories["restart_reason_history"] == ["time_control"]
    assert histories["pre_restart_reason_history"] == ["time_control"]
    assert histories["time_step_history"] == [pytest.approx(4.0)]
    assert histories["res0_history"] == [pytest.approx(5.0)]
    assert histories["res1_history"] == [pytest.approx(6.0)]
    assert histories["fsq_prev_history"] == [pytest.approx(7.0)]
    assert histories["bad_growth_streak_history"] == [8]
    assert histories["iter1_history"] == [9]
    assert histories["iter2_history"] == [10]
    assert histories["grad_rms_history"] == [pytest.approx(5.0)]
    assert histories["freeb_ivac_history"] == [3]
    assert histories["freeb_ivacskip_history"] == [0]
    assert histories["freeb_full_update_history"] == [1]
    lengths = {key: len(value) for key, value in histories.items()}
    assert set(lengths.values()) == {1}


def test_append_residual_iter_history_record_skips_free_boundary_channels_when_disabled():
    rec = _residual_iter_history_record(
        step=0.0,
        dt_eff=0.0,
        update_rms=0.0,
        w_curr=0.0,
        w_try=np.nan,
        w_try_ratio=np.nan,
        restart_path="converged",
        step_status="converged",
        restart_reason="none",
        pre_restart_reason="none",
        time_step=1.0,
        res0=0.0,
        res1=0.0,
        fsq_prev=0.0,
        bad_growth_streak=0,
        iter1=1,
        iter2=1,
        fsqr=0.0,
        fsqz=0.0,
        fsql=0.0,
        free_boundary_enabled=False,
    )
    histories = {
        "step_history": [],
        "dt_eff_history": [],
        "update_rms_history": [],
        "w_curr_history": [],
        "w_try_history": [],
        "w_try_ratio_history": [],
        "restart_path_history": [],
        "step_status_history": [],
        "restart_reason_history": [],
        "pre_restart_reason_history": [],
        "time_step_history": [],
        "res0_history": [],
        "res1_history": [],
        "fsq_prev_history": [],
        "bad_growth_streak_history": [],
        "iter1_history": [],
        "iter2_history": [],
        "grad_rms_history": [],
        "freeb_ivac_history": [],
        "freeb_ivacskip_history": [],
        "freeb_full_update_history": [],
    }

    _append_residual_iter_history_record(rec, free_boundary_enabled=False, **histories)

    assert histories["step_history"] == [0.0]
    assert histories["grad_rms_history"] == [0.0]
    assert histories["freeb_ivac_history"] == []
    assert histories["freeb_ivacskip_history"] == []
    assert histories["freeb_full_update_history"] == []


def _terminal_histories():
    return {
        "step_status_history": [],
        "restart_reason_history": [],
        "pre_restart_reason_history": [],
        "time_step_history": [],
        "res0_history": [],
        "res1_history": [],
        "fsq_prev_history": [],
        "bad_growth_streak_history": [],
        "iter1_history": [],
        "iter2_history": [],
        "grad_rms_history": [],
        "freeb_ivac_history": [],
        "freeb_ivacskip_history": [],
        "freeb_full_update_history": [],
        "freeb_nestor_reused_history": [],
        "freeb_nestor_solve_time_history": [],
        "freeb_nestor_sample_time_history": [],
    }


def test_append_residual_iter_terminal_history_records_free_boundary_channels():
    histories = _terminal_histories()

    _append_residual_iter_terminal_history(
        step_status="momentum",
        restart_reason="none",
        pre_restart_reason="none",
        time_step=0.75,
        res0=1.0,
        res1=0.5,
        fsq_prev=0.25,
        bad_growth_streak=2,
        iter1=3,
        iter2=4,
        fsqr=9.0,
        fsqz=16.0,
        fsql=0.0,
        free_boundary_enabled=True,
        freeb_ivac=2,
        freeb_ivacskip=0,
        freeb_reused=True,
        freeb_solve_time=0.125,
        freeb_sample_time=0.25,
        **histories,
    )

    assert histories["step_status_history"] == ["momentum"]
    assert histories["restart_reason_history"] == ["none"]
    assert histories["pre_restart_reason_history"] == ["none"]
    assert histories["time_step_history"] == [pytest.approx(0.75)]
    assert histories["res0_history"] == [pytest.approx(1.0)]
    assert histories["res1_history"] == [pytest.approx(0.5)]
    assert histories["fsq_prev_history"] == [pytest.approx(0.25)]
    assert histories["bad_growth_streak_history"] == [2]
    assert histories["iter1_history"] == [3]
    assert histories["iter2_history"] == [4]
    assert histories["grad_rms_history"] == [pytest.approx(5.0)]
    assert histories["freeb_ivac_history"] == [2]
    assert histories["freeb_ivacskip_history"] == [0]
    assert histories["freeb_full_update_history"] == [1]
    assert histories["freeb_nestor_reused_history"] == [1]
    assert histories["freeb_nestor_solve_time_history"] == [pytest.approx(0.125)]
    assert histories["freeb_nestor_sample_time_history"] == [pytest.approx(0.25)]
    assert {len(value) for value in histories.values()} == {1}


def test_append_residual_iter_terminal_history_skips_free_boundary_and_clamps_grad():
    histories = _terminal_histories()

    _append_residual_iter_terminal_history(
        step_status="rejected",
        restart_reason="bad_progress",
        pre_restart_reason="huge_initial_forces",
        time_step=0.5,
        res0=3.0,
        res1=2.0,
        fsq_prev=1.0,
        bad_growth_streak=7,
        iter1=8,
        iter2=9,
        fsqr=-10.0,
        fsqz=1.0,
        fsql=2.0,
        free_boundary_enabled=False,
        freeb_ivac=-1,
        freeb_ivacskip=99,
        freeb_reused=True,
        freeb_solve_time=9.0,
        freeb_sample_time=8.0,
        **histories,
    )

    assert histories["step_status_history"] == ["rejected"]
    assert histories["restart_reason_history"] == ["bad_progress"]
    assert histories["pre_restart_reason_history"] == ["huge_initial_forces"]
    assert histories["grad_rms_history"] == [0.0]
    assert histories["freeb_ivac_history"] == []
    assert histories["freeb_ivacskip_history"] == []
    assert histories["freeb_full_update_history"] == []
    assert histories["freeb_nestor_reused_history"] == []
    assert histories["freeb_nestor_solve_time_history"] == []
    assert histories["freeb_nestor_sample_time_history"] == []


def test_velocity_block_helpers_preserve_shape_dtype_and_scale():
    a = np.arange(6.0, dtype=np.float64).reshape(2, 3)
    b = np.arange(6, dtype=np.int32).reshape(2, 3)

    za, zb = _zero_velocity_blocks_like(a, b)
    assert np.asarray(za).shape == a.shape
    assert np.asarray(zb).shape == b.shape
    assert np.asarray(za).dtype == a.dtype
    assert np.asarray(zb).dtype == b.dtype
    np.testing.assert_allclose(np.asarray(za), 0.0)
    np.testing.assert_allclose(np.asarray(zb), 0.0)

    sa, sb = _scale_velocity_blocks(0.5, a, b)
    np.testing.assert_allclose(np.asarray(sa), 0.5 * a)
    np.testing.assert_allclose(np.asarray(sb), 0.5 * b)


def test_first_step_diagnostics_synthetic_default_and_axisymmetric_paths(monkeypatch):
    pytest.importorskip("jax")

    import vmec_jax.boundary as boundary_mod
    import vmec_jax.energy as energy_mod
    import vmec_jax.preconditioner_1d_jax as precond_mod
    import vmec_jax.solve as solve_mod
    import vmec_jax.static as static_mod
    import vmec_jax.vmec_forces as forces_mod
    import vmec_jax.vmec_residue as residue_mod
    import vmec_jax.vmec_tomnsp as tomnsp_mod

    s = np.array([0.0, 0.5, 1.0])
    modes = SimpleNamespace(m=np.array([0, 1]), n=np.array([0, 0]))
    shape = (3, 2, 1)
    ones = np.ones(shape)

    class DummyInData:
        scalars = {}
        indexed = {}

        def get_float(self, name, default=0.0):
            return {"DELT": 0.125, "TCON0": 1.75, "GAMMA": 0.0}.get(name, default)

        def get_bool(self, name, default=False):
            return {"LFORBAL": True, "LRFP": False}.get(name, default)

        def get_int(self, name, default=0):
            return {"NCURR": 0}.get(name, default)

    def make_static(*, lthreed: bool):
        cfg = SimpleNamespace(
            ns=3,
            mpol=2,
            ntor=0,
            nfp=1,
            ntheta=2,
            nzeta=1,
            lasym=False,
            lthreed=lthreed,
            lconm1=True,
        )
        return SimpleNamespace(cfg=cfg, s=s, modes=modes)

    def make_frzl(scale=1.0):
        return TomnspsRZL(
            frcc=scale * ones,
            frss=2.0 * scale * ones,
            fzsc=3.0 * scale * ones,
            fzcs=4.0 * scale * ones,
            flsc=5.0 * scale * ones,
            flcs=6.0 * scale * ones,
        )

    def fake_build_static(cfg, grid):
        return SimpleNamespace(cfg=cfg, s=s, modes=modes, trig_vmec=None, tomnsps_masks={"mask": True})

    bc = SimpleNamespace(
        guu=2.0 * ones,
        bsubu=ones,
        bsubv=2.0 * ones,
        jac=SimpleNamespace(r12=ones),
    )
    k = SimpleNamespace(bc=bc)

    monkeypatch.setattr(tomnsp_mod, "vmec_angle_grid", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(tomnsp_mod, "vmec_trig_tables", lambda **kwargs: SimpleNamespace(wint3_precond=np.ones((2, 1))))
    monkeypatch.setattr(static_mod, "build_static", fake_build_static)
    monkeypatch.setattr(
        energy_mod,
        "flux_profiles_from_indata",
        lambda indata, s, signgs: SimpleNamespace(
            chipf=np.array([0.0, 0.25, 0.5]),
            phips=np.array([9.0, 8.0, 7.0]),
            phipf=np.array([1.0, 1.5, 2.0]),
        ),
    )
    monkeypatch.setattr(boundary_mod, "boundary_from_indata", lambda indata, modes: SimpleNamespace(R_cos=np.array([2.0, 0.0])))
    monkeypatch.setattr(solve_mod, "_mass_half_mesh_from_indata", lambda **kwargs: np.array([0.0, 1.0, 2.0]))
    monkeypatch.setattr(solve_mod, "_pressure_half_mesh_from_indata", lambda **kwargs: np.array([0.0, 3.0, 4.0]))
    monkeypatch.setattr(solve_mod, "_icurv_full_mesh_from_indata", lambda **kwargs: np.array([0.0, 0.0, 0.0]))
    monkeypatch.setattr(
        solve_mod,
        "_vmec_force_flux_profiles",
        lambda **kwargs: (np.array([1.0, 1.0, 1.0]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])),
    )
    monkeypatch.setattr(forces_mod, "vmec_forces_rz_from_wout", lambda **kwargs: k)

    def fake_residual_internal_from_kernels(*args, **kwargs):
        assert kwargs["apply_lforbal"] is True
        assert kwargs["masks"] == {"mask": True}
        return make_frzl()

    monkeypatch.setattr(forces_mod, "vmec_residual_internal_from_kernels", fake_residual_internal_from_kernels)
    monkeypatch.setattr(residue_mod, "vmec_apply_scalxc_to_tomnsps", lambda *, frzl, s: frzl)
    monkeypatch.setattr(residue_mod, "vmec_apply_m1_constraints", lambda *, frzl, lconm1: frzl)
    monkeypatch.setattr(residue_mod, "vmec_zero_m1_zforce", lambda *, frzl, enabled: frzl)
    monkeypatch.setattr(residue_mod, "vmec_gcx2_from_tomnsps", lambda **kwargs: (np.array(1.0), np.array(2.0), np.array(3.0)))
    monkeypatch.setattr(
        residue_mod,
        "vmec_force_norms_from_bcovar_dynamic",
        lambda **kwargs: SimpleNamespace(r1=np.array(2.0), fnorm=np.array(3.0), fnormL=np.array(4.0)),
    )
    monkeypatch.setattr(residue_mod, "vmec_rz_norm_from_state", lambda **kwargs: np.array(5.0))
    monkeypatch.setattr(residue_mod, "vmec_scalxc_from_s", lambda **kwargs: np.array([1.0, 2.0, 3.0]))
    monkeypatch.setattr(residue_mod, "vmec_wint_from_trig", lambda trig, nzeta: np.ones((2, nzeta)))
    monkeypatch.setattr(precond_mod, "lambda_preconditioner", lambda **kwargs: 2.0 * ones)
    monkeypatch.setattr(
        precond_mod,
        "rz_preconditioner",
        lambda **kwargs: TomnspsRZL(
            frcc=np.full(shape, np.nan),
            frss=ones,
            fzsc=ones,
            fzcs=ones,
            flsc=ones,
            flcs=ones,
        ),
    )

    state0 = _state_from_value(0.5, ns=3, k=2)
    indata = DummyInData()

    default_diag = first_step_diagnostics(
        state0,
        make_static(lthreed=True),
        indata=indata,
        signgs=-1,
        step_size=0.25,
        include_constraint_force=True,
        use_axisymmetric_preconditioner=False,
    )
    assert default_diag["fsqr"] == pytest.approx(6.0)
    assert default_diag["fsql"] == pytest.approx(12.0)
    assert default_diag["time_step"] == pytest.approx(0.25)
    assert default_diag["frcc_u"].shape == shape
    np.testing.assert_allclose(default_diag["rz_scale"], 0.5)

    axis_diag = first_step_diagnostics(
        state0,
        make_static(lthreed=False),
        indata=indata,
        signgs=1,
        step_size=None,
        include_constraint_force=False,
        use_axisymmetric_preconditioner=True,
    )
    assert axis_diag["time_step"] == pytest.approx(0.125)
    np.testing.assert_allclose(axis_diag["frcc_u"], np.array([[[1.0], [0.5]], [[1.0], [0.5]], [[0.0], [0.0]]]))
