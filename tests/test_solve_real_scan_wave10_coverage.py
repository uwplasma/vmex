from __future__ import annotations

import numpy as np
import pytest

import vmec_jax.solve as solve


def _quiet_scan_env(monkeypatch):
    monkeypatch.setenv("VMEC_JAX_SCAN_PRINT", "0")
    monkeypatch.setenv("VMEC_JAX_SCAN_PRINT_CHUNKED", "0")
    monkeypatch.setenv("VMEC_JAX_SCAN_PREFLIGHT", "0")
    monkeypatch.setenv("VMEC_JAX_SCAN_EXTRA_ITERS", "0")
    monkeypatch.setenv("VMEC_JAX_SCAN_PRECOND_PRECOMPUTE", "0")
    monkeypatch.setenv("VMEC_JAX_TIMING", "0")


def _run_vmec2000_scan(state0, static, indata, **kwargs):
    params = dict(
        indata=indata,
        signgs=1,
        max_iter=1,
        step_size=float(indata.get_float("DELT", 1.0)),
        vmec2000_control=True,
        strict_update=True,
        backtracking=False,
        jit_forces=False,
        use_scan=True,
        scan_minimal_default=False,
        verbose=False,
        verbose_vmec2000_table=False,
    )
    params.update(kwargs)
    return solve.solve_fixed_boundary_residual_iter(state0, static, **params)


def test_vmec2000_scan_full_history_runs_fallback_decision(load_case_circular_tokamak, monkeypatch):
    pytest.importorskip("jax")

    _cfg, indata, static, _boundary, state0 = load_case_circular_tokamak
    _quiet_scan_env(monkeypatch)
    monkeypatch.setenv("VMEC_JAX_SCAN_LIGHT", "0")
    monkeypatch.setenv("VMEC_JAX_SCAN_MINIMAL", "0")
    monkeypatch.setenv("VMEC_JAX_SCAN_FALLBACK", "1")
    monkeypatch.setenv("VMEC_JAX_SCAN_FALLBACK_ITERS", "1")

    fallback_calls = []
    original_decision = solve._scan_fallback_decision

    def spy_fallback_decision(**kwargs):
        fallback_calls.append(kwargs)
        return original_decision(**kwargs)

    monkeypatch.setattr(solve, "_scan_fallback_decision", spy_fallback_decision)

    result = _run_vmec2000_scan(state0, static, indata)
    diag = result.diagnostics

    assert result.n_iter == 1
    assert result.w_history.shape == (1,)
    assert diag["use_scan"] is True
    assert diag["vmec2000_scan"] is True
    assert diag["scan_minimal"] is False
    assert diag["light_history"] is False
    assert np.asarray(diag["fsqr_full"]).shape == (1,)
    assert np.asarray(diag["fsqr1_history"]).shape == (1,)
    assert np.asarray(diag["time_step_history"]).shape == (1,)
    assert np.asarray(diag["min_tau_full"]).shape == (1,)
    assert np.isfinite(diag["final_fsqr"])
    assert len(fallback_calls) == 1
    assert fallback_calls[0]["max_iter"] == 1
    assert fallback_calls[0]["fallback_iters"] == 1


def test_vmec2000_scan_light_history_keeps_scalar_diagnostics(load_case_circular_tokamak, monkeypatch):
    pytest.importorskip("jax")

    _cfg, indata, static, _boundary, state0 = load_case_circular_tokamak
    _quiet_scan_env(monkeypatch)
    monkeypatch.setenv("VMEC_JAX_SCAN_LIGHT", "1")
    monkeypatch.setenv("VMEC_JAX_SCAN_MINIMAL", "0")
    monkeypatch.setenv("VMEC_JAX_SCAN_FALLBACK", "0")

    result = _run_vmec2000_scan(state0, static, indata, light_history=True)
    diag = result.diagnostics

    assert result.n_iter == 1
    assert diag["scan_minimal"] is False
    assert diag["light_history"] is True
    assert np.asarray(diag["fsqr_full"]).shape == (1,)
    assert np.asarray(diag["time_step_history"]).shape == (1,)
    assert np.asarray(diag["r00_history"]).shape == (1,)
    assert np.asarray(diag["bad_jacobian_full"]).shape == (1,)
    assert np.asarray(diag["fsqr1_history"]).shape == (0,)
    assert np.asarray(diag["zero_m1_history"]).shape == (0,)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"strict_update": False}, "strict_update=True"),
        ({"backtracking": True}, "backtracking=False"),
    ],
)
def test_vmec2000_scan_invalid_guards_raise(load_case_circular_tokamak, monkeypatch, overrides, match):
    pytest.importorskip("jax")

    _cfg, indata, static, _boundary, state0 = load_case_circular_tokamak
    _quiet_scan_env(monkeypatch)
    monkeypatch.setenv("VMEC_JAX_SCAN_PRINT", "0")
    monkeypatch.setenv("VMEC_JAX_SCAN_MINIMAL", "1")

    with pytest.raises(ValueError, match=match):
        _run_vmec2000_scan(state0, static, indata, **overrides)


def test_accelerated_scan_one_step_updates_state_and_histories(load_case_circular_tokamak, monkeypatch):
    pytest.importorskip("jax")

    _cfg, indata, static, _boundary, state0 = load_case_circular_tokamak
    _quiet_scan_env(monkeypatch)
    monkeypatch.setenv("VMEC_JAX_SCAN_LIGHT", "0")
    monkeypatch.setenv("VMEC_JAX_SCAN_MINIMAL", "0")
    monkeypatch.setenv("VMEC_JAX_SCAN_FALLBACK", "0")

    result = solve.solve_fixed_boundary_residual_iter(
        state0,
        static,
        indata=indata,
        signgs=1,
        max_iter=1,
        step_size=float(indata.get_float("DELT", 1.0)),
        vmec2000_control=False,
        strict_update=False,
        backtracking=False,
        auto_flip_force=False,
        use_restart_triggers=False,
        use_direct_fallback=False,
        precond_radial_alpha=0.0,
        precond_lambda_alpha=0.0,
        jit_forces=False,
        use_scan=True,
        scan_minimal_default=False,
        verbose=False,
        verbose_vmec2000_table=False,
    )

    diag = result.diagnostics
    assert result.n_iter == 1
    assert diag["use_scan"] is True
    assert diag["accelerated_scan"] is True
    assert diag.get("vmec2000_scan") is not True
    assert result.fsqr2_history.shape == (1,)
    assert result.fsqz2_history.shape == (1,)
    assert result.fsql2_history.shape == (1,)
    assert result.grad_rms_history.shape == (0,)
    assert result.step_history.shape == (0,)
    assert np.isfinite(result.w_history[-1])
    assert not np.allclose(np.asarray(result.state.Rcos), np.asarray(state0.Rcos))


def test_accelerated_scan_state_only_skips_history_outputs(load_case_circular_tokamak, monkeypatch):
    pytest.importorskip("jax")

    _cfg, indata, static, _boundary, state0 = load_case_circular_tokamak
    _quiet_scan_env(monkeypatch)
    monkeypatch.setenv("VMEC_JAX_SCAN_FALLBACK", "0")

    result = solve.solve_fixed_boundary_residual_iter(
        state0,
        static,
        indata=indata,
        signgs=1,
        max_iter=1,
        step_size=float(indata.get_float("DELT", 1.0)),
        vmec2000_control=False,
        strict_update=False,
        backtracking=False,
        auto_flip_force=False,
        use_restart_triggers=False,
        use_direct_fallback=False,
        precond_radial_alpha=0.0,
        precond_lambda_alpha=0.0,
        jit_forces=False,
        use_scan=True,
        state_only=True,
        verbose=False,
        verbose_vmec2000_table=False,
    )

    diag = result.diagnostics
    assert result.n_iter == 1
    assert diag["use_scan"] is True
    assert diag["accelerated_scan"] is True
    assert diag["state_only"] is True
    assert np.isfinite(diag["final_fsq_total"])
    assert result.w_history.shape == (0,)
    assert result.fsqr2_history.shape == (0,)
    assert result.fsqz2_history.shape == (0,)
    assert result.fsql2_history.shape == (0,)
    assert not np.allclose(np.asarray(result.state.Rcos), np.asarray(state0.Rcos))
