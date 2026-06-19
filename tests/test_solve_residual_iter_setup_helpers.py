from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from vmec_jax.solvers.fixed_boundary.residual.setup import (
    build_residual_cache_keys,
    free_boundary_pressure_edge_scale,
    grid_matches_vmec_static_grid,
    resolve_free_boundary_setup_policy,
)
from vmec_jax.solvers.fixed_boundary.residual.host_diagnostics import resolve_vmec2000_print_context
from vmec_jax.solvers.fixed_boundary.residual.ptau import (
    accepted_control_ptau_arrays,
    accepted_control_ptau_host_from_payload,
    maybe_dump_jacobian_terms,
    maybe_dump_ptau,
    ptau_minmax,
)
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
