from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import vmec_jax.discrete_adjoint as da
from vmec_jax._compat import has_jax
from vmec_jax.state import StateLayout, VMECState, pack_state, unpack_state


pytestmark = pytest.mark.skipif(not has_jax(), reason="discrete adjoint replay helpers require JAX")


def _state(values=None) -> VMECState:
    layout = StateLayout(ns=1, K=1, lasym=False)
    if values is None:
        values = np.arange(1.0, 7.0)
    blocks = [np.asarray([float(v)]).reshape(1, 1) for v in values]
    return VMECState(
        layout=layout,
        Rcos=blocks[0],
        Rsin=blocks[1],
        Zcos=blocks[2],
        Zsin=blocks[3],
        Lcos=blocks[4],
        Lsin=blocks[5],
    )


def _trace(state: VMECState, *, scale: float = 1.0, precond_jmax: int = 1, status: str = "momentum") -> dict:
    z = np.zeros((1, 1), dtype=float)
    return {
        "state_pre": state,
        "wout_like": "wout",
        "trig": "trig",
        "zero_m1": np.asarray(False),
        "precond_mats": "mats",
        "precond_jmax": precond_jmax,
        "lam_prec": "lam",
        "w_mode_mn": np.asarray([1.0]),
        "lambda_update_scale": float(scale),
        "dt_eff": 1.0,
        "b1": 1.0,
        "fac": 1.0,
        "force_scale": 1.0,
        "flip_sign": False,
        "time_step": 1.0,
        "fsq_prev_before": 1.0,
        "reset_inv_tau": False,
        "inv_tau_before": np.asarray([1.0]),
        "max_coeff_delta_rms_pre": 0.25,
        "max_update_rms_pre": 0.5,
        "apply_lforbal": False,
        "include_edge_residual": True,
        "apply_m1_constraints": True,
        "limit_update_rms": False,
        "limit_dt_from_force": False,
        "vmec2000_control": True,
        "divide_by_scalxc_for_update": False,
        "signgs": 1,
        "branch": "strict_update",
        "step_status": status,
        "restart_reason": "none",
        "restart_path": "momentum_accept",
        "vRcc_before": z,
        "vRss_before": z,
        "vRsc_before": z,
        "vRcs_before": z,
        "vZsc_before": z,
        "vZcs_before": z,
        "vZcc_before": z,
        "vZss_before": z,
        "vLsc_before": z,
        "vLcs_before": z,
        "vLcc_before": z,
        "vLss_before": z,
        "freeb_bsqvac_half": None,
        "freeb_pres_scale": None,
    }


def _tape(*traces, **overrides):
    defaults = dict(
        final_packed_state=np.zeros(6),
        packed_states=np.zeros((0, 6)),
        trace=da.concat_residual_iteration_traces([]),
        resume_states=(),
        step_traces=tuple(traces),
        stacked_step_traces=None,
        step_trace_static_flags=None,
        dynamic_initial_carry=None,
        dynamic_base_carries_stacked=None,
        diagnostics=None,
    )
    defaults.update(overrides)
    return da.ResidualCheckpointTape(**defaults)


def test_strict_update_one_step_threads_freeb_bsqvac_half_to_raw_residual(monkeypatch):
    from vmec_jax._compat import jnp

    state = _state()
    static = SimpleNamespace(cfg=SimpleNamespace())
    vac_edge = np.arange(4.0, dtype=float).reshape(2, 2)
    z = jnp.zeros((1, 1), dtype=float)
    captured = {}

    def fake_raw_force_residual_from_state(state_arg, static_arg, **kwargs):
        assert state_arg is state
        assert static_arg is static
        captured["freeb_bsqvac_half"] = kwargs.get("freeb_bsqvac_half")
        captured["freeb_pres_scale"] = kwargs.get("freeb_pres_scale")
        return {"k": SimpleNamespace(), "frzl": "raw-frzl"}

    def fake_preconditioned_force_channels_from_raw_forces(**kwargs):
        assert kwargs["frzl"] == "raw-frzl"
        return {
            "frcc_u": z,
            "frss_u": z,
            "fzsc_u": z,
            "fzcs_u": z,
            "flsc_u": z,
            "flcs_u": z,
        }

    def fake_strict_update_accepted_step(state_arg, static_arg, **kwargs):
        assert state_arg is state
        assert static_arg is static
        return {"state_post": state_arg}

    monkeypatch.setattr(da, "raw_force_residual_from_state", fake_raw_force_residual_from_state)
    monkeypatch.setattr(da, "preconditioned_force_channels_from_raw_forces", fake_preconditioned_force_channels_from_raw_forces)
    monkeypatch.setattr(da, "strict_update_accepted_step", fake_strict_update_accepted_step)

    out = da.strict_update_one_step_from_state(
        state,
        static,
        wout_like="wout",
        trig="trig",
        apply_lforbal=False,
        include_edge_residual=True,
        apply_m1_constraints=True,
        zero_m1=np.asarray(False),
        mats="mats",
        jmax=1,
        lam_prec="lam",
        w_mode_mn=np.asarray([1.0]),
        lambda_update_scale=1.0,
        dt_eff=1.0,
        b1=1.0,
        fac=1.0,
        force_scale=1.0,
        flip_sign=False,
        vRcc_before=z,
        vRss_before=z,
        vZsc_before=z,
        vZcs_before=z,
        vLsc_before=z,
        vLcs_before=z,
        limit_update_rms=False,
        need_update_rms=False,
        freeb_bsqvac_half=vac_edge,
        freeb_pres_scale=2.5,
    )

    np.testing.assert_allclose(np.asarray(captured["freeb_bsqvac_half"]), vac_edge)
    assert captured["freeb_pres_scale"] == 2.5
    assert out["step"]["state_post"] is state


def test_empty_tape_jvp_vjp_and_column_replay_are_identity():
    tape = _tape()
    tangent = np.asarray([1.0, 2.0, 3.0])
    columns = np.arange(6.0).reshape(2, 3)

    np.testing.assert_allclose(
        np.asarray(da.checkpoint_tape_state_jvp(tape=tape, static="static", initial_tangent=tangent)),
        tangent,
    )
    np.testing.assert_allclose(
        np.asarray(da.checkpoint_tape_state_vjp(tape=tape, static="static", final_cotangent=tangent)),
        tangent,
    )
    np.testing.assert_allclose(
        np.asarray(da.checkpoint_tape_state_jvp_columns(tape=tape, static="static", initial_tangents=columns)),
        columns,
    )


def test_state_jvp_and_vjp_replay_synthetic_linear_steps(monkeypatch):
    from vmec_jax._compat import jnp

    def fake_step(state, _static, **kwargs):
        x = pack_state(state)
        scale = jnp.asarray(kwargs["lambda_update_scale"], dtype=jnp.asarray(x).dtype)
        offset = jnp.arange(x.shape[0], dtype=jnp.asarray(x).dtype) + 10.0
        return {"step": {"state_post": unpack_state(scale * x + offset, state.layout)}}

    monkeypatch.setattr(da, "strict_update_one_step_from_state", fake_step)
    state0 = _state()
    tape = _tape(_trace(state0, scale=2.0), _trace(state0, scale=-0.5))
    tangent = np.asarray([1.0, -2.0, 3.0, -4.0, 5.0, -6.0])
    cotangent = np.asarray([2.0, 4.0, 6.0, 8.0, 10.0, 12.0])

    jvp = da.checkpoint_tape_state_jvp(
        tape=tape,
        static="static",
        initial_tangent=tangent,
        rebuild_preconditioner=False,
    )
    vjp = da.checkpoint_tape_state_vjp(
        tape=tape,
        static="static",
        final_cotangent=cotangent,
        rebuild_preconditioner=False,
    )

    np.testing.assert_allclose(np.asarray(jvp), -1.0 * tangent)
    np.testing.assert_allclose(np.asarray(vjp), -1.0 * cotangent)


def test_param_jvp_and_vjp_differentiate_synthetic_initial_guess(monkeypatch):
    from vmec_jax._compat import jnp

    layout = StateLayout(ns=1, K=1, lasym=False)

    def fake_apply_boundary_params(boundary, specs, params):
        assert boundary == "boundary"
        assert specs == "specs"
        return jnp.asarray(params)

    def fake_initial_guess_from_boundary(static, boundary_p, indata, *, vmec_project, axis_override):
        assert static == "static"
        assert indata == "indata"
        assert vmec_project is False
        assert axis_override == "axis"
        p = jnp.asarray(boundary_p)
        return VMECState(
            layout=layout,
            Rcos=p[0].reshape(1, 1),
            Rsin=(2.0 * p[1]).reshape(1, 1),
            Zcos=(p[0] + p[1]).reshape(1, 1),
            Zsin=jnp.zeros((1, 1), dtype=p.dtype),
            Lcos=p[1].reshape(1, 1),
            Lsin=(3.0 * p[0]).reshape(1, 1),
        )

    captured = {}

    def fake_state_jvp(*, tape, static, initial_tangent, rebuild_preconditioner):
        captured["jvp"] = (tape, static, np.asarray(initial_tangent), bool(rebuild_preconditioner))
        return jnp.asarray(initial_tangent) + 100.0

    def fake_state_vjp(*, tape, static, final_cotangent, rebuild_preconditioner):
        captured["vjp"] = (tape, static, np.asarray(final_cotangent), bool(rebuild_preconditioner))
        return jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    monkeypatch.setattr("vmec_jax.optimization.apply_boundary_params", fake_apply_boundary_params)
    monkeypatch.setattr("vmec_jax.init_guess.initial_guess_from_boundary", fake_initial_guess_from_boundary)
    monkeypatch.setattr(da, "checkpoint_tape_state_jvp", fake_state_jvp)
    monkeypatch.setattr(da, "checkpoint_tape_state_vjp", fake_state_vjp)

    tape = _tape()
    params = jnp.asarray([2.0, 5.0])
    params_tangent = jnp.asarray([0.25, -0.5])
    out_jvp = da.checkpoint_tape_param_jvp(
        tape=tape,
        static="static",
        boundary="boundary",
        indata="indata",
        specs="specs",
        params=params,
        axis_override="axis",
        params_tangent=params_tangent,
        vmec_project=False,
        rebuild_preconditioner=False,
    )
    out_vjp = da.checkpoint_tape_param_vjp(
        tape=tape,
        static="static",
        boundary="boundary",
        indata="indata",
        specs="specs",
        params=params,
        axis_override="axis",
        final_cotangent=np.arange(6.0),
        vmec_project=False,
        rebuild_preconditioner=True,
    )

    np.testing.assert_allclose(captured["jvp"][2], [0.25, -1.0, -0.25, 0.0, -0.5, 0.75])
    assert captured["jvp"][3] is False
    np.testing.assert_allclose(np.asarray(out_jvp), np.asarray(captured["jvp"][2]) + 100.0)
    np.testing.assert_allclose(captured["vjp"][2], np.arange(6.0))
    assert captured["vjp"][3] is True
    np.testing.assert_allclose(np.asarray(out_vjp), [22.0, 12.0])


def test_checkpoint_builder_optional_storage_paths_keep_shape_contracts(monkeypatch):
    state0 = _state()
    states = [_state(np.arange(2.0, 8.0)), _state(np.arange(3.0, 9.0))]
    calls = []

    def fake_replay_step(state, static, *, resume_state, solve_kwargs):
        calls.append((state, static, resume_state, dict(solve_kwargs)))
        idx = len(calls) - 1
        return SimpleNamespace(
            state=states[idx],
            diagnostics={
                "resume_state": {"idx": idx},
                "adjoint_step_trace": [_trace(state, scale=idx + 1.0)],
                "converged": idx == 1,
            },
        )

    monkeypatch.setattr(da, "replay_residual_checkpoint_step", fake_replay_step)
    monkeypatch.setattr(da, "_stack_replay_step_traces", lambda traces: ({"n": np.asarray([len(traces)])}, {"precond_jmax": 1}))

    tape = da.build_residual_checkpoint_tape(
        state0,
        "static",
        indata={"case": "synthetic"},
        signgs=1,
        max_iter=4,
        store_packed_states=False,
        store_trace=False,
        store_resume_states=False,
        light_history=True,
    )

    assert len(calls) == 2
    assert calls[0][3]["light_history"] is True
    assert calls[0][3]["resume_state_mode"] == "full"
    assert tape.packed_states.shape == (0, int(state0.layout.size))
    assert tape.trace.iter2.shape == (0,)
    assert tape.resume_states == ()
    assert len(tape.step_traces) == 2
    np.testing.assert_allclose(tape.final_packed_state, np.asarray(pack_state(states[-1])))
    assert tape.stacked_step_traces["n"][0] == 2


def test_replay_shape_helpers_detect_mismatches_and_missing_dynamic_layout():
    assert da._stacked_leading_axis_size({"a": np.zeros((2, 1)), "b": np.zeros((3, 1))}) is None
    assert da._stacked_leading_axis_size({"scalar": np.asarray(1.0)}) is None
    assert da._dynamic_basepoint_payload_shapes_match({"a": np.zeros((2, 1))}, (np.zeros((2, 1)),))
    assert not da._dynamic_basepoint_payload_shapes_match({"a": np.zeros((2, 1))}, (np.zeros((3, 1)),))

    carry = (np.zeros(6),) + tuple(np.zeros((1, 1)) for _ in range(14))
    trace = {"time_step": 1.0}
    with pytest.raises(ValueError, match="stored VMEC layout"):
        da._packed_dynamic_replay_step_from_carry(
            carry,
            trace,
            static=SimpleNamespace(),
            static_flags={},
            preconditioner_jmax_override=1,
        )


def test_static_trace_flags_compare_object_values_with_array_payloads():
    class Payload:
        def __init__(self, value):
            self.value = np.asarray(value)

        def __eq__(self, other):
            return isinstance(other, Payload) and np.array_equal(self.value, other.value)

    trace0 = _trace(_state(), precond_jmax=2)
    trace1 = _trace(_state(), precond_jmax=2)
    trace0["signgs"] = Payload([1.0, 2.0])
    trace1["signgs"] = Payload([1.0, 2.0])

    flags = da._static_flags_from_replay_step_traces((trace0, trace1))

    assert flags["precond_jmax"] == 2
    assert isinstance(flags["signgs"], Payload)
