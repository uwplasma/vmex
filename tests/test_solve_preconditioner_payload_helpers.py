from __future__ import annotations

import numpy as np
import pytest

from vmec_jax.solvers.fixed_boundary.residual import preconditioner_payload as payload_mod
from vmec_jax.solvers.fixed_boundary.residual.preconditioner_payload import (
    apply_vmec2000_preconditioner_runtime,
    host_preconditioned_residual_scalar_channels,
    jax_preconditioned_residual_scalar_channels,
    materialize_accepted_control_payload,
)
from vmec_jax.vmec_tomnsp import TomnspsRZL


class _FakeJax:
    calls = []

    @classmethod
    def device_get(cls, value):
        cls.calls.append(value)
        return value


def _tiny_frzl(value: float = 1.0) -> TomnspsRZL:
    arr = np.full((2, 1, 1), value, dtype=float)
    return TomnspsRZL(
        frcc=arr,
        frss=None,
        fzsc=2.0 * arr,
        fzcs=None,
        flsc=3.0 * arr,
        flcs=None,
        frsc=None,
        frcs=None,
        fzcc=None,
        fzss=None,
        flcc=None,
        flss=None,
    )


def _apply_runtime_kwargs(**overrides):
    frzl = _tiny_frzl()
    kwargs = dict(
        frzl=frzl,
        k="kernel",
        state="state",
        iter2=2,
        cfg=type("Cfg", (), {"lthreed": True, "lasym": False})(),
        s=np.asarray([0.0, 1.0]),
        delta_s=np.asarray(0.5),
        w_mode_mn=np.asarray([1.0]),
        lambda_update_scale=1.0,
        lambda_update_scale_j=np.asarray(1.0),
        lconm1=True,
        vmec2000_control=True,
        vmec2000_cache_valid=True,
        need_bcovar_update=False,
        cache_rz_norm=2.0,
        cache_f_norm1=0.5,
        host_update_assembly=False,
        use_fused_precond_output_scaling=False,
        scale_m1_rhs=True,
        adjoint_trace=False,
        adjoint_trace_mode="minimal",
        accepted_control_ptau_arrays=None,
        ptau_pshalf_jax=None,
        ptau_ohs_jax=None,
        preconditioner_use_precomputed_tridi=True,
        preconditioner_use_lax_tridi=False,
        timing_detail_enabled=False,
        timing_stats={"precond_apply": 0.0},
        perf_counter=iter([1.0, 1.25]).__next__,
        block_until_ready=lambda _x: None,
        refresh_preconditioner_cache_func=lambda _k, *, iter2: (
            np.asarray([2.0]),
            {"mats": True},
            1,
            True,
            False,
            True,
        ),
        scale_m1_precond_rhs_func=lambda frzl_in, _mats: frzl_in,
        rz_preconditioner_apply_func=lambda *, frzl_in, **_kwargs: frzl_in,
        rz_norm_func=lambda _state: np.asarray(2.0),
    )
    kwargs.update(overrides)
    return kwargs


def _host_from_payload(payload, *, device_get_floats):
    if payload == "unused":
        return 0.0, None, False
    fsq1, min_tau, max_tau = device_get_floats(*payload)
    return fsq1, (min_tau, max_tau), True


def test_host_preconditioned_residual_scalar_channels_uses_vmec2000_full_lambda_norm() -> None:
    out = host_preconditioned_residual_scalar_channels(
        gcr2_p=4.0,
        gcz2_p=8.0,
        gcl2_p=99.0,
        frzl_pre="frzl",
        frzl_pre_host="frzl-host",
        vmec2000_control=True,
        vmec2000_cache_valid=False,
        need_bcovar_update=True,
        cache_rz_norm=None,
        cache_f_norm1=None,
        state="state",
        delta_s=0.25,
        numpy_module=__import__("numpy"),
        rz_norm_np=lambda _state: 2.0,
        lambda_preconditioned_full_norm=lambda frzl, *, use_jax: 5.0 if frzl == "frzl-host" and not use_jax else 0.0,
        finite_float_or_zero=lambda value: 0.0 if value != value else float(value),
    )

    assert out.rz_norm == 2.0
    assert out.f_norm1 == 0.5
    assert out.fsqr1 == 2.0
    assert out.fsqz1 == 4.0
    assert out.fsql1 == 1.25
    assert out.fsq1 == 7.25


def test_jax_preconditioned_residual_scalar_channels_uses_cached_norm_and_safe_sum() -> None:
    import numpy as np

    out = jax_preconditioned_residual_scalar_channels(
        gcr2_p=np.asarray(4.0),
        gcz2_p=np.asarray(8.0),
        gcl2_p=np.asarray(9.0),
        frzl_pre="frzl",
        vmec2000_control=False,
        vmec2000_cache_valid=True,
        need_bcovar_update=False,
        cache_rz_norm=4.0,
        cache_f_norm1=0.25,
        state="state",
        delta_s=np.asarray(0.5),
        jnp_module=np,
        cached_or_current_f_norm1_jax=lambda **_kwargs: (np.asarray(4.0), np.asarray(0.25)),
        rz_norm_func=lambda _state: np.asarray(-1.0),
        lambda_preconditioned_full_norm=lambda *_args, **_kwargs: np.asarray(-1.0),
    )

    assert float(out.rz_norm) == 4.0
    assert float(out.f_norm1) == 0.25
    assert float(out.fsqr1_safe) == 1.0
    assert float(out.fsqz1_safe) == 2.0
    assert float(out.fsql1_safe) == 4.5
    assert float(out.fsq1) == 7.5


def test_materialize_accepted_control_payload_uses_existing_payload() -> None:
    stats = {"iteration_control_fsq1_payload_get": 0.0, "iteration_control_fsq1_direct_get": 0.0}

    out = materialize_accepted_control_payload(
        accepted_control_ptau_payload=(1.25, -0.5, 2.0),
        use_control_payload=True,
        fsq1_j=9.0,
        k="kernel",
        ptau_pshalf_jax="pshalf",
        ptau_ohs_jax="ohs",
        timing_enabled=True,
        timing_stats=stats,
        perf_counter=iter([10.0, 10.25]).__next__,
        jax_module=_FakeJax,
        device_get_floats=lambda *vals: tuple(float(v) for v in vals),
        accepted_control_ptau_host_from_payload=_host_from_payload,
        scan_math_kernel_arrays_from_k=lambda _k: (_k,),
        accepted_control_payload_jit=lambda: None,
    )

    assert out.fsq1 == 1.25
    assert out.accepted_control_ptau_host == (-0.5, 2.0)
    assert out.control_payload_used is True
    assert stats["iteration_control_fsq1_payload_get"] == pytest.approx(0.25)
    assert stats["iteration_control_fsq1_direct_get"] == 0.0


def test_materialize_accepted_control_payload_builds_payload_when_requested() -> None:
    stats = {"iteration_control_fsq1_payload_get": 0.0, "iteration_control_fsq1_direct_get": 0.0}
    payload_calls = []

    def payload_fn(fsq1_j, *args):
        payload_calls.append((fsq1_j, args))
        return (2.5, -1.0, 3.0)

    out = materialize_accepted_control_payload(
        accepted_control_ptau_payload=None,
        use_control_payload=True,
        fsq1_j=2.5,
        k="kernel",
        ptau_pshalf_jax="pshalf",
        ptau_ohs_jax="ohs",
        timing_enabled=True,
        timing_stats=stats,
        perf_counter=iter([20.0, 20.5]).__next__,
        jax_module=_FakeJax,
        device_get_floats=lambda *vals: tuple(float(v) for v in vals),
        accepted_control_ptau_host_from_payload=_host_from_payload,
        scan_math_kernel_arrays_from_k=lambda _k: ("ptau",),
        accepted_control_payload_jit=lambda: payload_fn,
    )

    assert payload_calls == [(2.5, ("ptau", "pshalf", "ohs"))]
    assert out.fsq1 == 2.5
    assert out.accepted_control_ptau_host == (-1.0, 3.0)
    assert out.control_payload_used is True
    assert stats["iteration_control_fsq1_payload_get"] == pytest.approx(0.5)
    assert stats["iteration_control_fsq1_direct_get"] == 0.0


def test_materialize_accepted_control_payload_falls_back_to_direct_device_get() -> None:
    _FakeJax.calls = []
    stats = {"iteration_control_fsq1_payload_get": 0.0, "iteration_control_fsq1_direct_get": 0.0}

    out = materialize_accepted_control_payload(
        accepted_control_ptau_payload="unused",
        use_control_payload=True,
        fsq1_j=7.0,
        k="kernel",
        ptau_pshalf_jax="pshalf",
        ptau_ohs_jax="ohs",
        timing_enabled=True,
        timing_stats=stats,
        perf_counter=iter([30.0, 30.1, 40.0, 40.2]).__next__,
        jax_module=_FakeJax,
        device_get_floats=lambda *vals: tuple(float(v) for v in vals),
        accepted_control_ptau_host_from_payload=_host_from_payload,
        scan_math_kernel_arrays_from_k=lambda _k: None,
        accepted_control_payload_jit=lambda: None,
    )

    assert out.fsq1 == 7.0
    assert out.accepted_control_ptau_host is None
    assert out.control_payload_used is False
    assert _FakeJax.calls == [7.0]
    assert stats["iteration_control_fsq1_payload_get"] == pytest.approx(0.1)
    assert stats["iteration_control_fsq1_direct_get"] == pytest.approx(0.2)


def test_apply_vmec2000_preconditioner_runtime_plain_path_keeps_raw_apply() -> None:
    out = apply_vmec2000_preconditioner_runtime(**_apply_runtime_kwargs())

    assert out.cache_update_trace is True
    assert out.frzl_lam_pre is not None
    assert out.update_blocks is None
    assert out.outputs_scaled is False
    assert out.fsq1_ready is False
    np.testing.assert_allclose(np.asarray(out.blocks.flsc), 6.0 * np.ones((2, 1, 1)))


def test_apply_vmec2000_preconditioner_runtime_output_payload_path(monkeypatch: pytest.MonkeyPatch) -> None:
    pre_blocks = tuple(np.asarray(float(i)) for i in range(12))
    update_blocks = tuple(np.asarray(10.0 + float(i)) for i in range(12))
    diag = tuple(np.asarray(v) for v in (1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.6))

    monkeypatch.setattr(
        payload_mod,
        "_preconditioner_output_payload_jit",
        lambda **_kwargs: lambda *_args: (pre_blocks, update_blocks, diag),
    )
    out = apply_vmec2000_preconditioner_runtime(
        **_apply_runtime_kwargs(use_fused_precond_output_scaling=True)
    )

    assert out.outputs_scaled is True
    assert out.fsq1_ready is True
    assert out.gcr2_p == pytest.approx(1.0)
    assert out.fsq1_safe == pytest.approx(0.6)
    assert out.blocks.frcc == pytest.approx(0.0)
    assert out.update_blocks.flss == pytest.approx(21.0)


def test_apply_vmec2000_preconditioner_runtime_fused_full_trace_materializes_raw(monkeypatch: pytest.MonkeyPatch) -> None:
    apply_calls = []
    pre_blocks = tuple(np.asarray(float(i)) for i in range(12))
    update_blocks = tuple(np.asarray(20.0 + float(i)) for i in range(12))
    diag = tuple(np.asarray(v) for v in (1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.6))

    monkeypatch.setattr(
        payload_mod,
        "_preconditioner_apply_payload_fused",
        lambda **_kwargs: (pre_blocks, update_blocks, diag, ("ptau",)),
    )

    def rz_apply(*, frzl_in, **kwargs):
        apply_calls.append(kwargs)
        return frzl_in

    out = apply_vmec2000_preconditioner_runtime(
        **_apply_runtime_kwargs(
            use_fused_precond_output_scaling=True,
            adjoint_trace=True,
            adjoint_trace_mode="full",
            refresh_preconditioner_cache_func=lambda _k, *, iter2: (
                np.asarray([2.0]),
                {"mats": True},
                1,
                False,
                False,
                False,
            ),
            rz_preconditioner_apply_func=rz_apply,
        )
    )

    assert out.outputs_scaled is True
    assert out.accepted_control_ptau_payload == ("ptau",)
    assert out.frzl_rz is not None
    assert out.frzl_lam_pre is None
    assert len(apply_calls) == 1
