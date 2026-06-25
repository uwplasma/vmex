from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax._compat import has_jax, jnp
from vmec_jax.solvers.fixed_boundary.diagnostics.axis_reset import (
    InitialAxisResetRuntimeCallbacks,
    bad_jacobian_from_tau_range,
    bad_jacobian_ptau_from_minmax,
    evaluate_initial_axis_reset,
    initial_force_physical_fsq,
    initial_axis_reset_runtime_decision,
    initial_axis_reset_runtime_update,
    reset_axis_from_boundary,
    run_initial_axis_reset_runtime,
    run_initial_axis_reset_setup,
)
from vmec_jax.solve import (
    _apply_vmec_lambda_axis_rules_to_state,
    _enforce_field_rows,
    _enforce_field_rows_np,
    _enforce_fixed_boundary_and_axis_np,
    _initial_axis_reset_decision,
    _merge_axis_reset_state,
    _write_axis_reset_dump,
)
from vmec_jax.state import StateLayout, VMECState


pytestmark = pytest.mark.skipif(not has_jax(), reason="solve axis helpers use JAX array operations")


def _state_from_base(base: np.ndarray, *, lasym: bool = False) -> VMECState:
    arr = np.asarray(base, dtype=float)
    layout = StateLayout(ns=int(arr.shape[0]), K=int(arr.shape[1]), lasym=lasym)
    return VMECState(
        layout=layout,
        Rcos=arr + 10.0,
        Rsin=arr + 20.0,
        Zcos=arr + 30.0,
        Zsin=arr + 40.0,
        Lcos=arr + 50.0,
        Lsin=arr + 60.0,
    )


def test_merge_axis_reset_state_uses_full_reset_fallback_and_cached_masks():
    state = _state_from_base(np.arange(6.0).reshape(2, 3))
    axis_state = _state_from_base(np.full((2, 3), 100.0))

    assert _merge_axis_reset_state(st=state, st_axis=axis_state, static=SimpleNamespace(), full_reset=True) is axis_state

    static_from_modes = SimpleNamespace(modes=SimpleNamespace(m=np.asarray([0, 1, 0])))
    merged = _merge_axis_reset_state(st=state, st_axis=axis_state, static=static_from_modes, full_reset=False)
    np.testing.assert_allclose(np.asarray(merged.Rcos)[:, [0, 2]], np.asarray(axis_state.Rcos)[:, [0, 2]])
    np.testing.assert_allclose(np.asarray(merged.Rcos)[:, 1], np.asarray(state.Rcos)[:, 1])
    np.testing.assert_allclose(np.asarray(merged.Lcos), np.asarray(state.Lcos))

    static_cached_mask = SimpleNamespace(
        modes=SimpleNamespace(m=np.asarray([9, 9, 9])),
        m_is_m0=np.asarray([0.0, 1.0, 0.0]),
    )
    cached = _merge_axis_reset_state(st=state, st_axis=axis_state, static=static_cached_mask, full_reset=False)
    np.testing.assert_allclose(np.asarray(cached.Zsin)[:, 1], np.asarray(axis_state.Zsin)[:, 1])
    np.testing.assert_allclose(np.asarray(cached.Zsin)[:, [0, 2]], np.asarray(state.Zsin)[:, [0, 2]])


def test_reset_axis_from_boundary_fallback_coefficients_preserve_non_axis_modes():
    class FakeIndata:
        def __init__(self, *, scalars, indexed):
            self.scalars = scalars
            self.indexed = indexed

    state = _state_from_base(np.arange(4.0).reshape(2, 2))
    static = SimpleNamespace(
        cfg=SimpleNamespace(ntor=1, ns=2),
        modes=SimpleNamespace(m=np.asarray([0, 1])),
    )
    indata = FakeIndata(scalars={"RAXIS_CC": [1.0], "ZAXIS_CS": [0.25]}, indexed={})

    def initial_guess_from_boundary(_static, _boundary, indata_local, *, dtype, infer_axis_if_missing):
        assert not infer_axis_if_missing
        raxis = float(indata_local.scalars["RAXIS_CC"][0])
        zaxis = float(indata_local.scalars["ZAXIS_CS"][0])
        layout = StateLayout(ns=2, K=2, lasym=False)
        rcos = np.full((2, 2), 110.0, dtype=dtype)
        zsin = np.full((2, 2), 140.0, dtype=dtype)
        rcos[0, 0] = raxis
        zsin[0, 0] = zaxis
        return VMECState(
            layout=layout,
            Rcos=rcos,
            Rsin=np.full((2, 2), 120.0, dtype=dtype),
            Zcos=np.full((2, 2), 130.0, dtype=dtype),
            Zsin=zsin,
            Lcos=np.full((2, 2), 150.0, dtype=dtype),
            Lsin=np.full((2, 2), 160.0, dtype=dtype),
        )

    def read_axis_coeffs(_indata):
        return {"RAXIS_CC": [1.0], "ZAXIS_CS": [0.25]}

    def recompute_axis_from_boundary(_static, _boundary, *, raxis_cc, zaxis_cs, signgs):
        assert signgs == 1
        return np.asarray([2.0, 3.0]), np.asarray([4.0, 5.0])

    out, coeffs = reset_axis_from_boundary(
        state,
        boundary_for_axis=object(),
        static=static,
        indata=indata,
        signgs=1,
        trig=object(),
        full_reset=False,
        zero_precond_diag=None,
        zero_tcon=None,
        constraint_active_false=False,
        compute_forces_iter_func=lambda *args, **kwargs: None,
        apply_vmec_lambda_axis_rules_func=lambda st: st,
        initial_guess_from_boundary_func=initial_guess_from_boundary,
        read_axis_coeffs_func=read_axis_coeffs,
        recompute_axis_from_state_vmec_func=lambda *args, **kwargs: None,
        recompute_axis_from_boundary_func=recompute_axis_from_boundary,
        axis_dump_dir="",
    )

    assert coeffs is not None
    np.testing.assert_allclose(coeffs[0], [2.0, 3.0])
    np.testing.assert_allclose(coeffs[3], [4.0, 5.0])
    np.testing.assert_allclose(np.asarray(out.Rcos)[:, 0], [2.0, 110.0])
    np.testing.assert_allclose(np.asarray(out.Rcos)[:, 1], np.asarray(state.Rcos)[:, 1])
    np.testing.assert_allclose(np.asarray(out.Lcos), np.asarray(state.Lcos))


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            dict(
                bad_jacobian_ptau=None,
                bad_jacobian_state=True,
                badjac_use_state=False,
                fsq_phys=2.0,
                axis_reset_fsq_min=0.0,
                force_axis_reset=False,
                axis_reset_always_3d=False,
                lthreed=False,
            ),
            (True, False, True),
        ),
        (
            dict(
                bad_jacobian_ptau=True,
                bad_jacobian_state=False,
                badjac_use_state=True,
                fsq_phys=2.0,
                axis_reset_fsq_min=0.0,
                force_axis_reset=False,
                axis_reset_always_3d=False,
                lthreed=True,
            ),
            (False, False, False),
        ),
        (
            dict(
                bad_jacobian_ptau=True,
                bad_jacobian_state=True,
                badjac_use_state=False,
                fsq_phys=0.25,
                axis_reset_fsq_min=1.0,
                force_axis_reset=False,
                axis_reset_always_3d=False,
                lthreed=True,
            ),
            (False, False, False),
        ),
        (
            dict(
                bad_jacobian_ptau=False,
                bad_jacobian_state=False,
                badjac_use_state=False,
                fsq_phys=None,
                axis_reset_fsq_min=1.0,
                force_axis_reset=False,
                axis_reset_always_3d=True,
                lthreed=True,
            ),
            (False, True, True),
        ),
        (
            dict(
                bad_jacobian_ptau=True,
                bad_jacobian_state=True,
                badjac_use_state=False,
                fsq_phys=np.inf,
                axis_reset_fsq_min=1.0,
                force_axis_reset=True,
                axis_reset_always_3d=False,
                lthreed=True,
                axis_reset_enabled=False,
            ),
            (False, True, False),
        ),
    ],
)
def test_initial_axis_reset_decision_branch_matrix(kwargs, expected):
    decision = _initial_axis_reset_decision(**kwargs)
    assert (decision.bad_jacobian, decision.force_reset, decision.reset) == expected


def test_initial_axis_reset_evaluation_skips_diagnostics_below_residual_floor():
    calls = {"ptau": 0, "state": 0}
    state = _state_from_base(np.ones((2, 2)))
    static = SimpleNamespace(
        cfg=SimpleNamespace(lthreed=True, lconm1=True),
        modes=SimpleNamespace(m=np.asarray([0, 1])),
    )

    result = evaluate_initial_axis_reset(
        axis_reset_enabled=True,
        norms=SimpleNamespace(r1=1.0, fnorm=1.0, fnormL=1.0),
        gcr2=0.1,
        gcz2=0.2,
        gcl2=0.3,
        k=object(),
        state=state,
        static=static,
        trig=SimpleNamespace(),
        s=np.asarray([0.0, 1.0]),
        badjac_use_state=True,
        ptau_tol=0.0,
        ptau_tol_rel=0.0,
        axis_reset_fsq_min=1.0,
        force_axis_reset=False,
        axis_reset_always_3d=False,
        vmec2000_control=True,
        lmove_axis=True,
        debug_enabled=False,
        state_check_on_missing_ptau=True,
        ptau_minmax_from_k_host=lambda _k: calls.__setitem__("ptau", calls["ptau"] + 1) or (-1.0, 1.0),
        vmec_half_mesh_jacobian_from_state_func=lambda **_kwargs: calls.__setitem__(
            "state", calls["state"] + 1
        ),
    )

    assert (result.decision.bad_jacobian, result.decision.force_reset, result.decision.reset) == (
        False,
        False,
        False,
    )
    assert result.fsq_phys == pytest.approx(0.6)
    assert result.bad_jacobian_ptau is None
    assert result.bad_jacobian_state is False
    assert calls == {"ptau": 0, "state": 0}


def test_initial_axis_reset_evaluation_forced_reset_keeps_diagnostics_below_residual_floor():
    calls = {"ptau": 0, "state": 0}
    state = _state_from_base(np.ones((2, 2)))
    static = SimpleNamespace(
        cfg=SimpleNamespace(lthreed=True, lconm1=True),
        modes=SimpleNamespace(m=np.asarray([0, 1])),
    )

    def ptau_minmax(_k):
        calls["ptau"] += 1
        return -1.0, 1.0

    result = evaluate_initial_axis_reset(
        axis_reset_enabled=True,
        norms=SimpleNamespace(r1=1.0, fnorm=1.0, fnormL=1.0),
        gcr2=0.1,
        gcz2=0.2,
        gcl2=0.3,
        k=object(),
        state=state,
        static=static,
        trig=SimpleNamespace(),
        s=np.asarray([0.0, 1.0]),
        badjac_use_state=False,
        ptau_tol=0.0,
        ptau_tol_rel=0.0,
        axis_reset_fsq_min=1.0,
        force_axis_reset=True,
        axis_reset_always_3d=False,
        vmec2000_control=True,
        lmove_axis=True,
        debug_enabled=False,
        state_check_on_missing_ptau=True,
        ptau_minmax_from_k_host=ptau_minmax,
        vmec_half_mesh_jacobian_from_state_func=lambda **_kwargs: calls.__setitem__(
            "state", calls["state"] + 1
        ),
    )

    assert (result.decision.bad_jacobian, result.decision.force_reset, result.decision.reset) == (
        False,
        True,
        True,
    )
    assert result.fsq_phys == pytest.approx(0.6)
    assert result.bad_jacobian_ptau is True
    assert result.bad_jacobian_state is False
    assert calls == {"ptau": 1, "state": 0}


def test_initial_axis_reset_setup_keeps_force_probe_when_no_reset():
    state = _state_from_base(np.ones((2, 2)))
    force_payload = (
        SimpleNamespace(),
        SimpleNamespace(),
        0.25,
        0.125,
        0.0625,
        np.asarray([1.0]),
        np.asarray([2.0]),
        SimpleNamespace(r1=1.0, fnorm=1.0, fnormL=1.0),
    )

    result = run_initial_axis_reset_setup(
        state=state,
        axis_reset_done=False,
        ijacob=0,
        state_checkpoint=state,
        velocities=(1, 2, 3, 4, 5, 6),
        res0=0.0,
        res1=0.0,
        prev_rz_fsq=1.0,
        vmec2000_control=True,
        lmove_axis=True,
        verbose=False,
        verbose_vmec2000_table=False,
        timing_enabled=False,
        timing_stats={},
        force_axis_reset=False,
        axis_reset_always_3d=False,
        axis_reset_fsq_min=1.0,
        badjac_use_state=False,
        static=SimpleNamespace(cfg=SimpleNamespace(lthreed=False, lconm1=True), modes=SimpleNamespace(m=np.asarray([0, 1]))),
        trig=SimpleNamespace(),
        s=np.asarray([0.0, 1.0]),
        zero_precond_diag=(np.zeros(2), np.zeros(2)),
        zero_tcon=np.zeros(2),
        compute_forces_iter_func=lambda *args, **kwargs: force_payload,
        reset_axis_from_boundary_func=lambda *args, **kwargs: pytest.fail("reset should not run"),
        zero_velocity_blocks_like_func=lambda *values: values,
        ptau_minmax_from_k_host_func=lambda _k: (1.0, 2.0),
        vmec_half_mesh_jacobian_from_state_func=lambda *args, **kwargs: pytest.fail("state probe unused"),
        print_axis_guess_func=lambda *args, **kwargs: None,
        axis_reset_coeffs_func=lambda: None,
        env_enabled_func=lambda _value: False,
        getenv_func=lambda _name, default="": default,
        perf_counter_func=lambda: 0.0,
        has_jax_func=lambda: False,
        block_until_ready_func=None,
        jnp_module=jnp,
    )

    assert result.reset_applied is False
    assert result.axis_reset_done is False
    assert result.force_probe is not None
    assert result.force_probe[0] is force_payload[0]
    assert result.force_probe[1] is force_payload[1]
    assert result.force_probe[2:5] == force_payload[2:5]
    assert result.force_probe[7] is force_payload[7]


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            dict(
                bad_jacobian=True,
                fsq_phys=0.25,
                axis_reset_fsq_min=1.0,
                force_axis_reset=False,
                axis_reset_always_3d=False,
                lthreed=True,
            ),
            (False, False, False, False),
        ),
        (
            dict(
                bad_jacobian=False,
                fsq_phys=np.inf,
                axis_reset_fsq_min=1.0,
                force_axis_reset=False,
                axis_reset_always_3d=False,
                lthreed=True,
            ),
            (False, True, False, True),
        ),
        (
            dict(
                bad_jacobian=False,
                fsq_phys=0.25,
                axis_reset_fsq_min=1.0,
                force_axis_reset=False,
                axis_reset_always_3d=True,
                lthreed=True,
            ),
            (False, False, True, True),
        ),
        (
            dict(
                bad_jacobian=True,
                fsq_phys=2.0,
                axis_reset_fsq_min=1.0,
                force_axis_reset=False,
                axis_reset_always_3d=False,
                lthreed=True,
                axis_reset_enabled=False,
            ),
            (True, False, False, False),
        ),
    ],
)
def test_initial_axis_reset_runtime_decision_preserves_in_loop_gate(kwargs, expected):
    decision = initial_axis_reset_runtime_decision(**kwargs)
    assert (
        decision.bad_jacobian,
        decision.huge_initial_forces,
        decision.force_reset,
        decision.reset,
    ) == expected


def test_initial_axis_reset_runtime_update_skips_outside_first_vmec2000_step():
    state = _state_from_base(np.zeros((2, 2)))
    result = initial_axis_reset_runtime_update(
        state=state,
        k=SimpleNamespace(),
        iter_idx=2,
        bad_jacobian=True,
        fsq_phys=2.0,
        axis_reset_done=False,
        lmove_axis=True,
        vmec2000_control=True,
        axis_reset_fsq_min=1.0,
        force_axis_reset=False,
        axis_reset_always_3d=False,
        lthreed=True,
        time_step=0.5,
        prev_rz_fsq_before=0.25,
        k_ndamp=3,
        reset_axis_from_boundary_func=lambda *_args, **_kwargs: pytest.fail("reset should not run"),
        host_axis_reset_update_func=lambda *_args, **_kwargs: pytest.fail("update should not run"),
    )

    assert result.state is state
    assert result.axis_reset_update is None
    assert not result.repeat_iteration
    assert not result.decision.reset


def test_initial_axis_reset_runtime_update_returns_reset_state_and_controller_payload():
    state = _state_from_base(np.zeros((2, 2)))
    reset_state = _state_from_base(np.ones((2, 2)))
    calls: dict[str, int] = {"reset": 0, "update": 0}

    def reset_axis(st, **kwargs):
        calls["reset"] += 1
        assert st is state
        assert kwargs["k_guess"].marker == "k"
        assert kwargs["full_reset"] is False
        assert kwargs["refine_axis_guess"] is False
        return reset_state

    def host_update(st, time_step, iter_idx, prev_rz_fsq_before, k_ndamp):
        calls["update"] += 1
        assert st is reset_state
        assert time_step == pytest.approx(0.5)
        assert iter_idx == 1
        assert prev_rz_fsq_before == pytest.approx(0.25)
        assert k_ndamp == 3
        return SimpleNamespace(marker="update")

    result = initial_axis_reset_runtime_update(
        state=state,
        k=SimpleNamespace(marker="k"),
        iter_idx=1,
        bad_jacobian=True,
        fsq_phys=2.0,
        axis_reset_done=False,
        lmove_axis=True,
        vmec2000_control=True,
        axis_reset_fsq_min=1.0,
        force_axis_reset=False,
        axis_reset_always_3d=False,
        lthreed=True,
        time_step=0.5,
        prev_rz_fsq_before=0.25,
        k_ndamp=3,
        reset_axis_from_boundary_func=reset_axis,
        host_axis_reset_update_func=host_update,
    )

    assert result.state is reset_state
    assert result.axis_reset_update.marker == "update"
    assert result.repeat_iteration
    assert result.decision.reset
    assert calls == {"reset": 1, "update": 1}


def test_run_initial_axis_reset_runtime_applies_callbacks_and_prints_axis_guess():
    state = _state_from_base(np.zeros((2, 2)))
    reset_state = _state_from_base(np.ones((2, 2)))
    calls: dict[str, int] = {"reset": 0, "host_update": 0, "apply": 0, "zero": 0, "guess": 0}
    messages: list[str] = []

    def reset_axis(_state, **_kwargs):
        calls["reset"] += 1
        return reset_state

    def host_update(_state, *_args):
        calls["host_update"] += 1
        return SimpleNamespace(marker="axis-update")

    def apply_controller(callback, payload):
        calls["apply"] += 1
        assert callback == "controller"
        assert payload.marker == "axis-update"

    def print_guess(raxis_cc, zaxis_cs):
        calls["guess"] += 1
        np.testing.assert_allclose(raxis_cc, [1.0])
        np.testing.assert_allclose(zaxis_cs, [0.0])

    result = run_initial_axis_reset_runtime(
        state=state,
        k=SimpleNamespace(),
        iter_idx=1,
        bad_jacobian=True,
        fsq_phys=2.0,
        axis_reset_done=False,
        lmove_axis=True,
        vmec2000_control=True,
        axis_reset_fsq_min=1.0,
        force_axis_reset=False,
        axis_reset_always_3d=False,
        lthreed=True,
        time_step=0.5,
        prev_rz_fsq_before=0.25,
        k_ndamp=3,
        verbose=True,
        verbose_vmec2000_table=True,
        callbacks=InitialAxisResetRuntimeCallbacks(
            reset_axis,
            host_update,
            apply_controller,
            "controller",
            lambda: calls.__setitem__("zero", calls["zero"] + 1),
            lambda: (np.array([1.0]), np.array([0.0]), np.array([0.0]), np.array([0.0])),
            print_guess,
        ),
        print_func=lambda message, **_kwargs: messages.append(message),
    )

    assert result.state is reset_state
    assert result.bad_jacobian
    assert result.reset
    assert result.repeat_iteration
    assert calls == {"reset": 1, "host_update": 1, "apply": 1, "zero": 1, "guess": 1}
    assert messages == [" INITIAL JACOBIAN CHANGED SIGN!", " TRYING TO IMPROVE INITIAL MAGNETIC AXIS GUESS"]


def test_initial_axis_reset_shared_bad_jacobian_helpers():
    norms = SimpleNamespace(r1=2.0, fnorm=3.0, fnormL=5.0)
    assert initial_force_physical_fsq(norms=norms, gcr2=0.5, gcz2=0.25, gcl2=0.1) == pytest.approx(5.0)
    assert initial_force_physical_fsq(norms=object(), gcr2=0.5, gcz2=0.25, gcl2=0.1) is None

    assert bad_jacobian_from_tau_range(min_tau=-1.0, max_tau=2.0)
    assert not bad_jacobian_from_tau_range(min_tau=-1.0e-4, max_tau=2.0, abs_tol=1.0e-3)

    assert bad_jacobian_ptau_from_minmax(
        ptau_min=-1.0,
        ptau_max=2.0,
        ptau_tol=0.0,
        ptau_tol_rel=0.0,
    )
    assert not bad_jacobian_ptau_from_minmax(
        ptau_min=-1.0e-4,
        ptau_max=2.0,
        ptau_tol=0.0,
        ptau_tol_rel=1.0e-3,
    )
    assert bad_jacobian_ptau_from_minmax(
        ptau_min=None,
        ptau_max=2.0,
        ptau_tol=0.0,
        ptau_tol_rel=0.0,
    ) is None


def test_write_axis_reset_dump_disabled_short_and_success_paths(tmp_path):
    assert not _write_axis_reset_dump(
        axis_dump_dir=" ",
        ns=3,
        ntor=1,
        used_state_guess=False,
        raxis_cc=[1.0, 2.0],
        raxis_cs=[0.0, 0.0],
        zaxis_cc=[0.0, 0.0],
        zaxis_cs=[0.0, 0.0],
    )

    assert not _write_axis_reset_dump(
        axis_dump_dir=tmp_path,
        ns=3,
        ntor=2,
        used_state_guess=False,
        raxis_cc=[1.0, 2.0],
        raxis_cs=[0.0, 0.0],
        zaxis_cc=[0.0, 0.0],
        zaxis_cs=[0.0, 0.0],
    )

    assert _write_axis_reset_dump(
        axis_dump_dir=tmp_path,
        ns=5,
        ntor=1,
        used_state_guess=True,
        raxis_cc=[1.0, 2.0],
        raxis_cs=[0.1, 0.2],
        zaxis_cc=[3.0, 4.0],
        zaxis_cs=[0.3, 0.4],
    )
    text = (tmp_path / "axis_reset_ns5.dat").read_text(encoding="utf-8")
    assert "# used_state_guess=1" in text
    assert "0" in text and "1" in text


def test_apply_vmec_lambda_axis_rules_host_device_and_disabled_paths():
    state = _state_from_base(np.arange(9.0).reshape(3, 3))

    assert (
        _apply_vmec_lambda_axis_rules_to_state(
            state,
            enforce_vmec_lambda_axis=False,
            host_update_assembly=True,
            idx00=1,
        )
        is state
    )

    host_invalid = _apply_vmec_lambda_axis_rules_to_state(
        state,
        enforce_vmec_lambda_axis=True,
        host_update_assembly=True,
        idx00=10,
    )
    np.testing.assert_allclose(np.asarray(host_invalid.Lcos), np.asarray(state.Lcos))

    host = _apply_vmec_lambda_axis_rules_to_state(
        state,
        enforce_vmec_lambda_axis=True,
        host_update_assembly=True,
        idx00=1,
    )
    np.testing.assert_allclose(np.asarray(host.Lcos)[:, 1], 0.0)
    np.testing.assert_allclose(np.asarray(host.Lsin)[:, 1], 0.0)

    device = _apply_vmec_lambda_axis_rules_to_state(
        state,
        enforce_vmec_lambda_axis=True,
        host_update_assembly=False,
        idx00=2,
    )
    np.testing.assert_allclose(np.asarray(device.Lcos)[:, 2], 0.0)
    np.testing.assert_allclose(np.asarray(device.Lsin)[:, 2], 0.0)


def test_enforce_field_rows_jax_handles_empty_single_and_combined_constraints():
    empty = _enforce_field_rows(np.empty((0, 2)), axis_mask=np.ones(2), edge_row=np.arange(2.0))
    assert np.asarray(empty).shape == (0, 2)

    single = _enforce_field_rows(
        np.asarray([[9.0, 9.0, 9.0]]),
        axis_mask=np.asarray([1.0, 0.0, 1.0]),
        edge_row=np.asarray([3.0, 4.0, 5.0]),
    )
    np.testing.assert_allclose(np.asarray(single), [[3.0, 0.0, 5.0]])

    single_zero = _enforce_field_rows(
        np.asarray([[9.0, 9.0, 9.0]]),
        axis_mask=np.asarray([1.0, 0.0, 1.0]),
        edge_row=np.asarray([3.0, 4.0, 5.0]),
        zero_axis=True,
    )
    np.testing.assert_allclose(np.asarray(single_zero), [[0.0, 0.0, 0.0]])

    base = np.arange(12.0).reshape(4, 3)
    combined = _enforce_field_rows(
        base,
        axis_mask=np.asarray([1.0, 0.0, 1.0]),
        edge_row=np.asarray([100.0, 101.0, 102.0]),
    )
    np.testing.assert_allclose(np.asarray(combined)[0], [0.0, 0.0, 2.0])
    np.testing.assert_allclose(np.asarray(combined)[-1], [100.0, 101.0, 102.0])
    np.testing.assert_allclose(np.asarray(_enforce_field_rows(base, zero_axis=True))[0], 0.0)
    np.testing.assert_allclose(np.asarray(_enforce_field_rows(base, edge_row=[7.0, 8.0, 9.0]))[-1], [7.0, 8.0, 9.0])


def test_enforce_field_rows_np_handles_empty_single_and_combined_constraints():
    empty = _enforce_field_rows_np(np.empty((0, 2)), axis_mask=np.ones(2), edge_row=np.arange(2.0))
    assert empty.shape == (0, 2)

    single = _enforce_field_rows_np(
        np.asarray([[9.0, 9.0, 9.0]]),
        axis_mask=np.asarray([1.0, 0.0, 1.0]),
        edge_row=np.asarray([3.0, 4.0, 5.0]),
    )
    np.testing.assert_allclose(single, [[3.0, 0.0, 5.0]])

    single_zero = _enforce_field_rows_np(
        np.asarray([[9.0, 9.0, 9.0]]),
        edge_row=np.asarray([3.0, 4.0, 5.0]),
        zero_axis=True,
    )
    np.testing.assert_allclose(single_zero, [[0.0, 0.0, 0.0]])

    base = np.arange(12.0).reshape(4, 3)
    combined = _enforce_field_rows_np(
        base,
        axis_mask=np.asarray([1.0, 0.0, 1.0]),
        edge_row=np.asarray([100.0, 101.0, 102.0]),
    )
    np.testing.assert_allclose(combined[0], [0.0, 0.0, 2.0])
    np.testing.assert_allclose(combined[-1], [100.0, 101.0, 102.0])
    np.testing.assert_allclose(_enforce_field_rows_np(base, zero_axis=True)[0], 0.0)


def test_enforce_fixed_boundary_and_axis_np_respects_toggles_and_precomputed_mask():
    state = _state_from_base(np.arange(12.0).reshape(4, 3))
    static = SimpleNamespace(modes=SimpleNamespace(m=np.asarray([9, 9, 9])))
    mask = np.asarray([1.0, 0.0, 1.0])

    constrained = _enforce_fixed_boundary_and_axis_np(
        state,
        static,
        edge_Rcos=np.asarray([100.0, 101.0, 102.0]),
        edge_Rsin=np.asarray([110.0, 111.0, 112.0]),
        edge_Zcos=np.asarray([120.0, 121.0, 122.0]),
        edge_Zsin=np.asarray([130.0, 131.0, 132.0]),
        enforce_axis=True,
        enforce_edge=True,
        enforce_lambda_axis=True,
        idx00=2,
        precomputed_axis_mask=mask,
    )
    np.testing.assert_allclose(constrained.Rcos[0], [10.0, 0.0, 12.0])
    np.testing.assert_allclose(constrained.Rsin[-1], [110.0, 111.0, 112.0])
    np.testing.assert_allclose(constrained.Zsin[-1], [130.0, 131.0, 132.0])
    np.testing.assert_allclose(constrained.Lcos[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(constrained.Lcos[:, 2], 0.0)
    np.testing.assert_allclose(constrained.Lsin[:, 2], 0.0)

    unconstrained = _enforce_fixed_boundary_and_axis_np(
        state,
        static,
        edge_Rcos=np.asarray([100.0, 101.0, 102.0]),
        edge_Rsin=np.asarray([110.0, 111.0, 112.0]),
        edge_Zcos=np.asarray([120.0, 121.0, 122.0]),
        edge_Zsin=np.asarray([130.0, 131.0, 132.0]),
        enforce_axis=False,
        enforce_edge=False,
        enforce_lambda_axis=False,
        idx00=None,
        precomputed_axis_mask=mask,
    )
    np.testing.assert_allclose(unconstrained.Rcos, np.asarray(state.Rcos))
    np.testing.assert_allclose(unconstrained.Lcos, np.asarray(state.Lcos))
