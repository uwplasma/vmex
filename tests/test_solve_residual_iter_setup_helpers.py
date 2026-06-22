from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from vmec_jax.solvers.fixed_boundary.residual.setup import (
    build_residual_cache_keys,
    build_residual_ptau_bindings,
    build_residual_profile_setup,
    free_boundary_pressure_edge_scale,
    grid_matches_vmec_static_grid,
    resolve_free_boundary_setup_policy,
)
from vmec_jax.solvers.fixed_boundary.residual.host_diagnostics import (
    dump_residual_evolve_trace,
    evaluate_vmec2000_time_control,
    print_compact_converged_status,
    print_compact_physical_residual_status,
    print_compact_residual_iteration_update_status,
    print_residual_iteration_update_status,
    residual_update_rms_for_print,
    resolve_vmec2000_print_context,
    sample_vmec_iteration_scalars,
)
from vmec_jax.solvers.fixed_boundary.residual.update import ResidualVelocityBlocks
from vmec_jax.solvers.fixed_boundary.residual.ptau import (
    accepted_control_ptau_arrays,
    accepted_control_ptau_host_from_payload,
    maybe_dump_jacobian_terms,
    maybe_dump_ptau,
    ptau_minmax,
)
from vmec_jax.solvers.fixed_boundary.residual.policy import vmec2000_time_control_decision
from vmec_jax.solvers.fixed_boundary.residual.scan_adapters import (
    ScanConvergencePredicate,
    ScanDeviceRuntime,
    ScanTimeControlDumper,
    ScanVmec2000PrintContext,
    scan_m1_preconditioner_rhs,
)
from vmec_jax.solvers.fixed_boundary.residual.state_setup import build_residual_state_setup
from vmec_jax.config import VMECConfig
from vmec_jax.state import StateLayout, VMECState
from vmec_jax.static import build_static


def _cfg(**updates):
    values = {
        "lfreeb": False,
        "nvacskip": 0,
        "nfp": 2,
        "ntor": 3,
        "mpol": 4,
        "ns": 8,
        "lasym": False,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def _small_static_state():
    cfg = VMECConfig(
        ns=4,
        mpol=2,
        ntor=1,
        nfp=2,
        lasym=False,
        lthreed=True,
        lconm1=True,
        ntheta=8,
        nzeta=4,
    )
    static = build_static(cfg)
    layout = StateLayout(ns=int(cfg.ns), K=int(static.modes.K), lasym=False)
    base = np.arange(layout.ns * layout.K, dtype=float).reshape(layout.ns, layout.K)
    state = VMECState(
        layout=layout,
        Rcos=base + 1.0,
        Rsin=base + 2.0,
        Zcos=base + 3.0,
        Zsin=base + 4.0,
        Lcos=base + 5.0,
        Lsin=base + 6.0,
    )
    return static, state


def test_build_residual_profile_setup_records_profile_and_trig_subtimings() -> None:
    static, state0 = _small_static_state()
    timings = {"setup_profile_data": 0.0, "setup_trig_tables": 0.0}
    clock_values = iter([10.0, 12.5, 20.0, 23.0])
    calls = []

    def build_profiles(**kwargs):
        calls.append(("profiles", kwargs))
        assert kwargs["prefer_host_default_profiles"] is True
        return SimpleNamespace(wout_like="wout-like")

    def resolve_trig(**kwargs):
        calls.append(("trig", kwargs))
        assert kwargs["wout_like"] == "wout-like"
        return "trig-tables"

    wout_like, trig = build_residual_profile_setup(
        indata=SimpleNamespace(),
        static=static,
        s=np.linspace(0.0, 1.0, int(static.cfg.ns)),
        signgs=1,
        idx00=0,
        state0=state0,
        state0_has_tracer=False,
        host_update_assembly=False,
        host_profile_setup=False,
        build_wout_like_profiles_func=build_profiles,
        resolve_residual_trig_func=resolve_trig,
        vmec_trig_tables_func=lambda **_kwargs: "unused",
        tree_has_tracer_func=lambda _value: False,
        jnp_module=np,
        setup_phase_timings=timings,
        timing_enabled=True,
        perf_counter_func=lambda: next(clock_values),
    )

    assert wout_like == "wout-like"
    assert trig == "trig-tables"
    assert [kind for kind, _ in calls] == ["profiles", "trig"]
    assert np.isclose(timings["setup_profile_data"], 2.5)
    assert np.isclose(timings["setup_trig_tables"], 3.0)


def test_build_residual_state_setup_host_path_caches_constants_and_enforces_edge() -> None:
    static, state0 = _small_static_state()
    edge_Rcos = np.full(int(static.modes.K), 10.0)
    edge_Rsin = np.full(int(static.modes.K), 20.0)
    edge_Zcos = np.full(int(static.modes.K), 30.0)
    edge_Zsin = np.full(int(static.modes.K), 40.0)

    setup = build_residual_state_setup(
        state0=state0,
        static=static,
        s=np.linspace(0.0, 1.0, int(static.cfg.ns)),
        edge_Rcos=edge_Rcos,
        edge_Rsin=edge_Rsin,
        edge_Zcos=edge_Zcos,
        edge_Zsin=edge_Zsin,
        free_boundary_enabled=False,
        host_update_assembly=True,
        setup_host_enforce=True,
        idx00=0,
        mpol=int(static.cfg.mpol),
        nrange=2 * int(static.cfg.ntor) + 1,
        state0_dtype=np.asarray(state0.Rcos).dtype,
        apply_lambda_axis_rules=lambda state: state,
        tree_has_tracer=lambda _value: False,
        has_jax_func=lambda: False,
    )

    np.testing.assert_allclose(setup.state.Rcos[-1], edge_Rcos)
    np.testing.assert_allclose(setup.state.Rsin[-1], edge_Rsin)
    np.testing.assert_allclose(setup.state.Zcos[-1], edge_Zcos)
    np.testing.assert_allclose(setup.state.Zsin[-1], edge_Zsin)
    assert setup.precomputed_axis_mask_np.shape == (int(static.modes.K),)
    assert setup.jnp_zero_m1_0 is None
    assert setup.zeros_coeff_np.shape == (int(static.cfg.ns), int(static.cfg.mpol), 2 * int(static.cfg.ntor) + 1)
    assert setup.zeros_dR_np.shape == np.asarray(state0.Rcos).shape
    assert float(setup.delta_s) == 1.0 / 3.0


def test_ptau_wrapper_dispatches_and_preserves_call_time_dump_arguments() -> None:
    ptau_context = SimpleNamespace(pshalf_np=np.asarray([1.0, 2.0]), ohs_scalar=3.0, pshalf_jax="pj", ohs_jax="oj", s="s")
    host_calls = []
    jax_calls = []

    host_result = ptau_minmax(
        "kernel",
        ptau_context=ptau_context,
        has_jax_func=lambda: False,
        compute_jit="jit",
        pshalf_from_s_jax=lambda *_args, **_kwargs: None,
        ptau_minmax_host_func=lambda *args, **kwargs: host_calls.append((args, kwargs)) or (-1.0, 2.0),
        ptau_minmax_jax_func=lambda *args, **kwargs: jax_calls.append((args, kwargs)) or (-3.0, 4.0),
    )
    assert host_result == (-1.0, 2.0)
    assert host_calls[0][1]["compute_jit"] == "jit"
    assert host_calls[0][1]["pshalf_jax"] == "pj"
    assert not jax_calls

    jax_result = ptau_minmax(
        "kernel",
        ptau_context=ptau_context,
        has_jax_func=lambda: True,
        compute_jit="jit",
        pshalf_from_s_jax=lambda *_args, **_kwargs: "ps",
        ptau_minmax_host_func=lambda *args, **kwargs: host_calls.append((args, kwargs)) or (-1.0, 2.0),
        ptau_minmax_jax_func=lambda *args, **kwargs: jax_calls.append((args, kwargs)) or (-3.0, 4.0),
    )
    assert jax_result == (-3.0, 4.0)
    assert jax_calls[-1][1]["s"] == "s"

    assert accepted_control_ptau_arrays("k", kernel_arrays_from_k=lambda _k: None) is None
    assert accepted_control_ptau_arrays("k", kernel_arrays_from_k=lambda _k: (np.ones((1,)),)) is None
    arrays = accepted_control_ptau_arrays("k", kernel_arrays_from_k=lambda _k: (np.ones((2,)),))
    assert arrays is not None
    fsq1, ptau_host, used = accepted_control_ptau_host_from_payload(
        (np.asarray(1.25), np.asarray(-0.5), np.asarray(0.75)),
        device_get_floats=lambda *args: tuple(float(np.asarray(arg)) for arg in args),
    )
    assert used is True
    assert fsq1 == 1.25
    assert ptau_host == (-0.5, 0.75)
    assert accepted_control_ptau_host_from_payload(
        (np.asarray(1.0),),
        device_get_floats=lambda *args: tuple(float(np.asarray(arg)) for arg in args),
    ) == (None, None, False)

    jacobian_dump = {}
    maybe_dump_jacobian_terms(k="k", s="s", iter_idx=7, dump_func=lambda **kwargs: jacobian_dump.update(kwargs))
    assert jacobian_dump == {"k": "k", "s": "s", "iter_idx": 7}

    ptau_dump = {}
    maybe_dump_ptau(
        iter_idx=8,
        ptau_min=-1.0,
        ptau_max=2.0,
        tau_min_state=None,
        tau_max_state=3.0,
        badjac_ptau=True,
        badjac_state=False,
        badjac_used=True,
        mode="host",
        label="probe",
        getenv=lambda name, default: {"VMEC_JAX_DUMP_PTAU": "1", "VMEC_JAX_DUMP_DIR": "/tmp/dump"}.get(
            name, default
        ),
        dump_func=lambda **kwargs: ptau_dump.update(kwargs),
    )
    assert ptau_dump["dump_ptau_env"] == "1"
    assert ptau_dump["dump_dir"] == "/tmp/dump"
    assert ptau_dump["label"] == "probe"


def test_residual_ptau_bindings_disable_jit_for_concrete_host_assembly() -> None:
    host_calls = []
    jax_calls = []

    _, minmax_host, minmax, _ = build_residual_ptau_bindings(
        s=np.asarray([0.0, 0.5, 1.0]),
        has_jax_value=True,
        s_has_tracer=False,
        pshalf_from_s_np_func=lambda _s: np.asarray([0.25, 0.75]),
        pshalf_from_s_jax_func=lambda *_args, **_kwargs: "pshalf_jax",
        build_context_func=lambda s, **kwargs: SimpleNamespace(
            s=s,
            pshalf_np=kwargs["pshalf_from_s_np"](s),
            ohs_scalar=2.0,
            pshalf_jax="pshalf_jax",
            ohs_jax="ohs_jax",
        ),
        compute_jit_func="jit",
        ptau_minmax_host_helper=lambda *args, **kwargs: host_calls.append((args, kwargs)) or (-1.0, 1.0),
        ptau_minmax_helper=ptau_minmax,
        scan_ptau_minmax_host_func=lambda *args, **kwargs: (-2.0, 2.0),
        scan_ptau_minmax_jax_func=lambda *args, **kwargs: jax_calls.append((args, kwargs)) or (-3.0, 3.0),
        accepted_control_ptau_arrays_helper=accepted_control_ptau_arrays,
        scan_kernel_arrays_from_k_func=lambda _k: None,
        has_jax_func=lambda: False,
        host_update_assembly=True,
    )

    assert minmax_host("k") == (-1.0, 1.0)
    assert host_calls[-1][1]["compute_jit"] is None

    assert minmax("k") == (-2.0, 2.0)
    assert not jax_calls


def test_residual_ptau_bindings_keep_jit_for_traced_assembly() -> None:
    host_calls = []

    _, minmax_host, _minmax, _ = build_residual_ptau_bindings(
        s=np.asarray([0.0, 0.5, 1.0]),
        has_jax_value=True,
        s_has_tracer=True,
        pshalf_from_s_np_func=lambda _s: np.asarray([0.25, 0.75]),
        pshalf_from_s_jax_func=lambda *_args, **_kwargs: "pshalf_jax",
        build_context_func=lambda s, **_kwargs: SimpleNamespace(
            s=s,
            pshalf_np=None,
            ohs_scalar=None,
            pshalf_jax="pshalf_jax",
            ohs_jax="ohs_jax",
        ),
        compute_jit_func="jit",
        ptau_minmax_host_helper=lambda *args, **kwargs: host_calls.append((args, kwargs)) or (-1.0, 1.0),
        ptau_minmax_helper=ptau_minmax,
        scan_ptau_minmax_host_func=lambda *args, **kwargs: (-2.0, 2.0),
        scan_ptau_minmax_jax_func=lambda *args, **kwargs: (-3.0, 3.0),
        accepted_control_ptau_arrays_helper=accepted_control_ptau_arrays,
        scan_kernel_arrays_from_k_func=lambda _k: None,
        has_jax_func=lambda: False,
        host_update_assembly=True,
    )

    assert minmax_host("k") == (-1.0, 1.0)
    assert host_calls[-1][1]["compute_jit"] == "jit"


def test_resolve_vmec2000_print_context_forwards_rows_and_cadence() -> None:
    rows = []

    context = resolve_vmec2000_print_context(
        cfg=SimpleNamespace(lasym=True),
        indata=SimpleNamespace(get_int=lambda _name, _default: 5),
        verbose=True,
        vmec2000_control=True,
        verbose_vmec2000_table=True,
        getenv=lambda name, default: {"VMEC_JAX_SCAN_PRINT": "0", "VMEC_JAX_NSTEP_OVERRIDE": "3"}.get(
            name, default
        ),
        resolve_debug_print_config=lambda **_kwargs: SimpleNamespace(
            mode="debug_print",
            ordered=False,
            print_live=False,
        ),
        resolve_nstep_screen=lambda **kwargs: int(kwargs["override_env"]),
        emit_iter_row=lambda **kwargs: rows.append(kwargs),
        should_print_row=lambda **kwargs: kwargs["iter_idx"] % kwargs["nstep_screen"] == 0,
        print_row=lambda **_kwargs: None,
    )

    assert context.nstep_screen == 3
    assert context.should_print(6, 10)
    assert not context.should_print(7, 10)
    context.print_iter_row(
        iter_idx=6,
        fsqr=1.0,
        fsqz=2.0,
        fsql=3.0,
        fsqr1=4.0,
        fsqz1=5.0,
        fsql1=6.0,
        delt0r=0.9,
        r00=1.25,
        w_mhd=7.0,
        z00=-0.5,
    )
    assert rows[0]["iter_idx"] == 6
    assert rows[0]["lasym"]
    assert rows[0]["scan_print_mode"] == "debug_print"


def test_sample_vmec_iteration_scalars_preserves_host_vmec2000_rounding() -> None:
    scalars = sample_vmec_iteration_scalars(
        need_scalar=True,
        k=SimpleNamespace(
            pr1_even=np.asarray([[[1.23456]]]),
            pz1_even=np.asarray([[[-0.56789]]]),
        ),
        state=SimpleNamespace(Rcos=np.zeros((1, 1)), Zcos=np.zeros((1, 1))),
        norms_current=SimpleNamespace(wb=2.0, wp=3.0),
        m0_mask=np.asarray([True]),
        lasym=True,
        host_update_assembly=True,
        vmec2000_control=True,
        gamma=2.0,
        twopi=2.0,
        previous_r00=9.0,
        previous_z00=8.0,
        previous_wb=7.0,
        previous_wp=6.0,
        tree_has_tracer=lambda _value: False,
        device_get_floats=lambda *_values: (_ for _ in ()).throw(AssertionError("device path not expected")),
        jnp_module=np,
    )

    assert scalars.r00 == 1.235
    assert scalars.z00 == -0.5679
    assert scalars.wb == 2.0
    assert scalars.wp == 3.0
    assert scalars.w_vmec == 20.0


def test_sample_vmec_iteration_scalars_falls_back_to_state_and_previous_values() -> None:
    state = SimpleNamespace(
        Rcos=np.asarray([[1.0, 2.0, 4.0]]),
        Zcos=np.asarray([[3.0, 5.0, 7.0]]),
    )
    fallback = sample_vmec_iteration_scalars(
        need_scalar=True,
        k=SimpleNamespace(),
        state=state,
        norms_current=SimpleNamespace(wb=0.25, wp=0.5),
        m0_mask=np.asarray([True, False, True]),
        lasym=True,
        host_update_assembly=True,
        vmec2000_control=False,
        gamma=2.0,
        twopi=2.0,
        previous_r00=9.0,
        previous_z00=8.0,
        previous_wb=7.0,
        previous_wp=6.0,
        tree_has_tracer=lambda _value: False,
        device_get_floats=lambda *_values: (_ for _ in ()).throw(AssertionError("device path not expected")),
        jnp_module=np,
    )
    assert fallback.r00 == 5.0
    assert fallback.z00 == 10.0
    assert fallback.w_vmec == 3.0

    previous = sample_vmec_iteration_scalars(
        need_scalar=False,
        k=SimpleNamespace(),
        state=state,
        norms_current=SimpleNamespace(wb=0.25, wp=0.5),
        m0_mask=np.asarray([True, False, True]),
        lasym=True,
        host_update_assembly=True,
        vmec2000_control=False,
        gamma=2.0,
        twopi=2.0,
        previous_r00=9.0,
        previous_z00=8.0,
        previous_wb=7.0,
        previous_wp=6.0,
        tree_has_tracer=lambda _value: False,
        device_get_floats=lambda *_values: (_ for _ in ()).throw(AssertionError("device path not expected")),
        jnp_module=np,
    )
    assert previous.r00 == 9.0
    assert previous.z00 == 8.0
    assert previous.w_vmec == 52.0


def test_sample_vmec_iteration_scalars_uses_device_callback_when_host_path_is_unavailable() -> None:
    seen = []

    scalars = sample_vmec_iteration_scalars(
        need_scalar=True,
        k=SimpleNamespace(
            pr1_even=np.asarray([[[1.5]]]),
            pz1_even=np.asarray([[[2.5]]]),
        ),
        state=SimpleNamespace(Rcos=np.zeros((1, 1)), Zcos=np.zeros((1, 1))),
        norms_current=SimpleNamespace(wb=3.0, wp=4.0),
        m0_mask=np.asarray([True]),
        lasym=False,
        host_update_assembly=False,
        vmec2000_control=False,
        gamma=3.0,
        twopi=2.0,
        previous_r00=0.0,
        previous_z00=0.0,
        previous_wb=0.0,
        previous_wp=0.0,
        tree_has_tracer=lambda _value: True,
        device_get_floats=lambda *values: seen.append(values) or tuple(float(np.asarray(value)) for value in values),
        jnp_module=np,
    )

    assert len(seen) == 1
    assert scalars.r00 == 1.5
    assert scalars.z00 == 0.0
    assert scalars.w_vmec == 20.0


def test_residual_status_helpers_route_compact_and_vmec2000_rows() -> None:
    messages = []

    assert print_compact_physical_residual_status(
        verbose=True,
        vmec2000_control=False,
        verbose_vmec2000_table=False,
        iter_idx=2,
        fsqr=1.0,
        fsqz=2.0,
        fsql=3.0,
        include_edge=True,
        print_func=lambda message, **_kwargs: messages.append(message),
    )
    assert "iter=002" in messages[-1]

    assert not print_compact_converged_status(
        verbose=True,
        vmec2000_control=True,
        verbose_vmec2000_table=True,
        fsqr=1.0,
        fsqz=2.0,
        fsql=3.0,
        target=4.0,
        print_func=lambda message, **_kwargs: messages.append(message),
    )
    assert "converged" not in messages[-1]

    assert print_compact_residual_iteration_update_status(
        verbose=True,
        vmec2000_control=False,
        verbose_vmec2000_table=False,
        precond_diag_floats=lambda: (4.0, 5.0, 6.0),
        iter_idx=3,
        dt_eff=0.25,
        update_rms=0.125,
        step_status="accepted",
        print_func=lambda message, **_kwargs: messages.append(message),
    )
    assert "step_status=accepted" in messages[-1]

    assert residual_update_rms_for_print(
        verbose=False,
        strict_update=True,
        update_rms_j=np.asarray(12.0),
        update_rms=34.0,
    ) == 0.0
    assert residual_update_rms_for_print(
        verbose=True,
        strict_update=True,
        update_rms_j=np.asarray(12.0),
        update_rms=34.0,
    ) == 12.0
    assert residual_update_rms_for_print(
        verbose=True,
        strict_update=False,
        update_rms_j=np.asarray(12.0),
        update_rms=34.0,
    ) == 34.0

    vmec_rows = []
    assert not print_residual_iteration_update_status(
        verbose=True,
        vmec2000_control=True,
        verbose_vmec2000_table=True,
        should_print_vmec2000=lambda _iter_idx, _max_iter: False,
        print_vmec2000_iter_row=lambda **kwargs: vmec_rows.append(kwargs),
        precond_diag_floats=lambda: (7.0, 8.0, 9.0),
        iter_idx=4,
        max_iter=10,
        compact_iter_idx=3,
        fsqr=1.0,
        fsqz=2.0,
        fsql=3.0,
        dt_eff=0.0,
        update_rms=0.0,
        time_step=0.9,
        r00=1.25,
        z00=-0.5,
        w_mhd=6.0,
        step_status="rejected",
    )
    assert not vmec_rows

    assert print_residual_iteration_update_status(
        verbose=True,
        vmec2000_control=True,
        verbose_vmec2000_table=True,
        should_print_vmec2000=lambda _iter_idx, _max_iter: False,
        print_vmec2000_iter_row=lambda **kwargs: vmec_rows.append(kwargs),
        precond_diag_floats=lambda: (7.0, 8.0, 9.0),
        iter_idx=4,
        max_iter=10,
        compact_iter_idx=3,
        fsqr=1.0,
        fsqz=2.0,
        fsql=3.0,
        dt_eff=0.0,
        update_rms=0.0,
        time_step=0.9,
        r00=1.25,
        z00=-0.5,
        w_mhd=6.0,
        step_status="converged",
        force_vmec2000_row=True,
    )
    assert vmec_rows[-1]["fsqr1"] == 7.0
    assert vmec_rows[-1]["r00"] == 1.25


def test_evaluate_vmec2000_time_control_emits_initial_checkpoint_sequence() -> None:
    traces = []
    checkpoints = []
    restarts = []

    decision = evaluate_vmec2000_time_control(
        iter2=1,
        iter1=1,
        fsq_prev=0.5,
        fsq0_curr=0.25,
        fsq0_prev=0.75,
        res0=-1.0,
        res1=-1.0,
        bad_jacobian=False,
        vmec2000_fact=2.0,
        time_step=0.9,
        time_control_decision=vmec2000_time_control_decision,
        dump_time_control_trace=lambda **kwargs: traces.append(kwargs),
        maybe_dump_checkpoint=lambda **kwargs: checkpoints.append(kwargs),
        maybe_dump_time_control=lambda **kwargs: restarts.append(kwargs),
    )

    assert decision.initialized
    assert decision.store_checkpoint
    assert not decision.restart
    assert [row["stage"] for row in traces] == ["init", "pre", "checkpoint"]
    assert len(checkpoints) == 2
    assert not restarts
    assert checkpoints[0]["fsq"] == 0.5
    assert traces[-1]["irst"] == 1


def test_evaluate_vmec2000_time_control_emits_restart_sequence() -> None:
    traces = []
    checkpoints = []
    restarts = []

    decision = evaluate_vmec2000_time_control(
        iter2=5,
        iter1=1,
        fsq_prev=0.5,
        fsq0_curr=0.25,
        fsq0_prev=0.75,
        res0=0.1,
        res1=0.1,
        bad_jacobian=True,
        vmec2000_fact=2.0,
        time_step=0.9,
        time_control_decision=vmec2000_time_control_decision,
        dump_time_control_trace=lambda **kwargs: traces.append(kwargs),
        maybe_dump_checkpoint=lambda **kwargs: checkpoints.append(kwargs),
        maybe_dump_time_control=lambda **kwargs: restarts.append(kwargs),
    )

    assert decision.restart
    assert decision.irst == 2
    assert decision.pre_restart_reason == "bad_jacobian"
    assert [row["stage"] for row in traces] == ["pre", "restart"]
    assert traces[-1]["irst"] == 2
    assert traces[-1]["fsq0"] == 0.75
    assert not checkpoints
    assert restarts == [
        {
            "iter_idx": 5,
            "fsq": 0.5,
            "fsq0": 0.75,
            "res0": 0.1,
            "res1": 0.1,
            "time_step": 0.9,
        }
    ]


def test_dump_residual_evolve_trace_maps_velocity_and_force_blocks() -> None:
    rows = []
    velocities = ResidualVelocityBlocks(*(f"v{idx}" for idx in range(12)))
    forces = ResidualVelocityBlocks(*(f"f{idx}" for idx in range(12)))

    dump_residual_evolve_trace(
        dump_evolve_trace=lambda **kwargs: rows.append(kwargs),
        iter2=7,
        iter1=3,
        stage="post",
        fsq1=1.0,
        fsq_prev=2.0,
        time_step=0.9,
        dtau=0.1,
        b1=0.8,
        fac=0.7,
        state="state",
        velocities=velocities,
        forces=forces,
    )

    row = rows[0]
    assert row["iter2"] == 7
    assert row["stage"] == "post"
    assert row["state_val"] == "state"
    assert row["vRcc_val"] == "v0"
    assert row["vLss_val"] == "v11"
    assert row["frcc_val"] == "f0"
    assert row["flss_val"] == "f11"


def test_scan_adapter_contexts_delegate_runtime_and_scan_contracts() -> None:
    ready_calls = []

    runtime = ScanDeviceRuntime(
        scan_timing_enabled=True,
        stats={"scan_device_s": 0.0},
        perf_counter=lambda: 3.5,
        block_until_ready=lambda value: ready_calls.append(value) or value,
        tree_map=lambda func, value: func(value),
        record_ready=lambda **kwargs: ready_calls.append(("record", kwargs)),
    )
    assert runtime.block_value("payload") == "payload"
    assert ready_calls[0] == "payload"
    assert runtime.ready(1.0, "device", cache_status="hit") == "device"
    assert ready_calls[-1][0] == "record"
    assert ready_calls[-1][1]["cache_status"] == "hit"

    print_context = ScanVmec2000PrintContext(
        nstep_screen=5,
        lasym=False,
        verbose=True,
        vmec2000_control=True,
        verbose_vmec2000_table=True,
    )
    assert print_context.should_print(5, 12)
    assert not print_context.should_print(6, 12)

    dumper = ScanTimeControlDumper(
        enabled=False,
        timecontrol_callback=None,
        timecontrol_path=None,
        jax_module=None,
        jnp_module=np,
    )
    assert int(dumper(cond=True, stage_id=1, iter2=2, iter1=1, fsq=1.0, fsq0=1.0, res0=1.0, res1=1.0, time_step=0.9, irst=1)) == 0

    predicate = ScanConvergencePredicate(
        ftol=1.0e-3,
        fsq_total_target=None,
        converged_func=lambda fsqr, fsqz, fsql, *, ftol, fsq_total_target: (fsqr + fsqz + fsql) <= ftol,
    )
    assert predicate(1.0e-4, 2.0e-4, 3.0e-4)
    assert not predicate(1.0e-3, 2.0e-3, 3.0e-3)

    call_args = {}
    assert (
        scan_m1_preconditioner_rhs(
            "forces",
            {"mat": "value"},
            cfg=SimpleNamespace(lconm1=False, mpol=7),
            scale_m1_precond_rhs_from_mats=lambda *args, **kwargs: call_args.update(
                {"args": args, "kwargs": kwargs}
            )
            or "scaled",
        )
        == "scaled"
    )
    assert call_args["args"] == ("forces", {"mat": "value"})
    assert call_args["kwargs"] == {"lconm1": False, "mpol": 7, "host_update_assembly": False}


def test_grid_matches_vmec_static_grid_requires_same_coordinates() -> None:
    grid = SimpleNamespace(nfp=2, theta=np.asarray([0.0, 0.5]), zeta=np.asarray([0.0, 0.25]))
    same = SimpleNamespace(nfp=2, theta=np.asarray([0.0, 0.5]), zeta=np.asarray([0.0, 0.25]))
    changed_nfp = SimpleNamespace(nfp=3, theta=same.theta, zeta=same.zeta)
    changed_theta = SimpleNamespace(nfp=2, theta=np.asarray([0.0, 0.6]), zeta=same.zeta)
    broken = SimpleNamespace(nfp=2, theta=object(), zeta=same.zeta)

    assert grid_matches_vmec_static_grid(grid, same)
    assert not grid_matches_vmec_static_grid(grid, changed_nfp)
    assert not grid_matches_vmec_static_grid(grid, changed_theta)
    assert not grid_matches_vmec_static_grid(broken, same)


def test_build_residual_cache_keys_delegates_hash_and_edge_signatures() -> None:
    static = SimpleNamespace(
        cfg=SimpleNamespace(mpol=4, ntor=2, ntheta=8, nzeta=6, nfp=3, ns=5, lasym=False),
        modes=SimpleNamespace(m=np.asarray([0, 1]), n=np.asarray([0, 1])),
        grid=SimpleNamespace(theta=np.asarray([0.0, 0.5]), zeta=np.asarray([0.0, 0.25])),
    )
    wout_like = SimpleNamespace(
        nfp=3,
        mpol=4,
        ntor=2,
        lasym=False,
        signgs=-1,
        phipf=np.asarray([1.0, 2.0]),
        phips=np.asarray([0.0, 1.0]),
        chipf=np.asarray([0.0, 0.5]),
        pres=np.asarray([0.0, 0.1]),
        icurv=np.asarray([0.0, 0.2]),
    )

    def fake_hash(value):
        return ("hash", tuple(np.asarray(value).shape), float(np.asarray(value).sum()))

    keys = build_residual_cache_keys(
        static=static,
        wout_like=wout_like,
        edge_Rcos=np.asarray([1.0]),
        edge_Rsin=np.asarray([2.0]),
        edge_Zcos=np.asarray([3.0]),
        edge_Zsin=np.asarray([4.0]),
        constraint_tcon0=1.25,
        hash_array_bytes_func=fake_hash,
        edge_signature_key_func=lambda *arrays: ("sig", len(arrays)),
        edge_value_key_func=lambda *arrays: ("val", sum(float(np.asarray(a).sum()) for a in arrays)),
    )

    assert keys.static_key[:7] == (4, 2, 8, 6, 3, 5, False)
    assert keys.wout_key[:5] == (3, 4, 2, False, -1)
    assert keys.wout_key[-1] == 1.25
    assert keys.edge_signature_key == ("sig", 4)
    assert keys.edge_value_key == ("val", 10.0)


def test_free_boundary_setup_policy_disables_scan_and_resolves_direct_provider() -> None:
    policy = resolve_free_boundary_setup_policy(
        _cfg(lfreeb=True, nvacskip=4),
        external_field_provider_kind="Coils",
        use_scan=True,
        freeb_couple_env="1",
        freeb_sample_env="yes",
        jit_strict_update_env="off",
        backend_name="cpu",
        host_update_assembly=True,
        cpu_work_limit_env="1000",
    )

    assert policy.free_boundary_enabled
    assert policy.free_boundary_provider_kind == "coils"
    assert policy.direct_free_boundary_provider
    assert policy.freeb_nvacskip == 4
    assert policy.freeb_nvskip0 == 4
    assert policy.freeb_couple_edge
    assert not policy.use_scan
    assert policy.freeb_sample_external
    assert not policy.jit_strict_update_enabled


def test_free_boundary_pressure_edge_scale_uses_last_half_mesh_pressure_ratio() -> None:
    calls = []

    def fake_eval_profiles(_indata, s_values):
        s_arr = np.asarray(s_values, dtype=float)
        calls.append(float(s_arr[0]))
        return {"pressure": 4.0 * s_arr}

    scale = free_boundary_pressure_edge_scale(
        free_boundary_enabled=True,
        indata={"profile": "synthetic"},
        s=np.linspace(0.0, 1.0, 5),
        eval_profiles_func=fake_eval_profiles,
    )

    np.testing.assert_allclose(scale, 1.0 / 0.875)
    np.testing.assert_allclose(calls, [0.875, 1.0])


def test_free_boundary_pressure_edge_scale_handles_disabled_zero_and_errors() -> None:
    assert (
        free_boundary_pressure_edge_scale(
            free_boundary_enabled=False,
            indata={"profile": "synthetic"},
            s=np.linspace(0.0, 1.0, 5),
            eval_profiles_func=lambda _indata, _s: {"pressure": np.asarray([1.0])},
        )
        is None
    )
    assert (
        free_boundary_pressure_edge_scale(
            free_boundary_enabled=True,
            indata=None,
            s=np.linspace(0.0, 1.0, 5),
            eval_profiles_func=lambda _indata, _s: {"pressure": np.asarray([1.0])},
        )
        is None
    )

    zero_edge = free_boundary_pressure_edge_scale(
        free_boundary_enabled=True,
        indata={"profile": "synthetic"},
        s=np.linspace(0.0, 1.0, 5),
        eval_profiles_func=lambda _indata, _s: {"pressure": np.asarray([0.0])},
    )
    assert zero_edge == 0.0

    failed = free_boundary_pressure_edge_scale(
        free_boundary_enabled=True,
        indata={"profile": "synthetic"},
        s=np.linspace(0.0, 1.0, 5),
        eval_profiles_func=lambda _indata, _s: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert failed is None


def test_free_boundary_setup_policy_auto_strict_update_matches_cpu_gpu_defaults() -> None:
    small_cpu_host = resolve_free_boundary_setup_policy(
        _cfg(lfreeb=False, ns=8, mpol=4, ntor=3),
        external_field_provider_kind=None,
        use_scan=False,
        freeb_couple_env="0",
        freeb_sample_env="0",
        jit_strict_update_env="auto",
        backend_name="cpu",
        host_update_assembly=True,
        cpu_work_limit_env="100",
    )
    assert not small_cpu_host.free_boundary_enabled
    assert small_cpu_host.free_boundary_provider_kind == ""
    assert not small_cpu_host.direct_free_boundary_provider
    assert small_cpu_host.freeb_nvacskip == 1
    assert not small_cpu_host.freeb_couple_edge
    assert not small_cpu_host.freeb_sample_external
    assert not small_cpu_host.jit_strict_update_enabled
    assert small_cpu_host.update_work == 8 * 4 * 4

    large_cpu_device = resolve_free_boundary_setup_policy(
        _cfg(lfreeb=False, ns=20, mpol=6, ntor=5, lasym=True),
        external_field_provider_kind="mgrid",
        use_scan=False,
        freeb_couple_env="1",
        freeb_sample_env="1",
        jit_strict_update_env="auto",
        backend_name="cpu",
        host_update_assembly=False,
        cpu_work_limit_env="100",
    )
    assert large_cpu_device.update_work == 20 * 6 * 11
    assert large_cpu_device.jit_strict_update_enabled

    gpu = resolve_free_boundary_setup_policy(
        _cfg(lfreeb=False, ns=1, mpol=1, ntor=0),
        external_field_provider_kind="direct_coils",
        use_scan=False,
        freeb_couple_env="1",
        freeb_sample_env="1",
        jit_strict_update_env="auto",
        backend_name="gpu",
        host_update_assembly=True,
        cpu_work_limit_env="not-an-int",
    )
    assert gpu.jit_strict_update_enabled
    assert gpu.cpu_work_limit == 1000
