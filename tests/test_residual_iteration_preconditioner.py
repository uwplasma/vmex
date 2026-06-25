from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from vmec_jax.solvers.fixed_boundary.residual.iteration_preconditioner import (
    apply_residual_iteration_preconditioner,
    resolve_preconditioned_residual_scalars,
)
from vmec_jax.solvers.fixed_boundary.residual.payload_blocks import ForceBlocks


def _force_blocks(value: float) -> ForceBlocks:
    arr = np.full((2, 2), float(value))
    return ForceBlocks(*(arr + idx for idx in range(12)))


def _common_kwargs(**overrides):
    timings: list[str] = []
    kwargs = dict(
        use_vmec2000_preconditioner=False,
        frzl=_force_blocks(1.0),
        k=SimpleNamespace(),
        state=SimpleNamespace(),
        iter2=1,
        cfg=SimpleNamespace(lthreed=True, lasym=False),
        static=SimpleNamespace(cfg=SimpleNamespace(lconm1=True)),
        s=np.array([0.0, 1.0]),
        delta_s=0.5,
        w_mode_mn=np.ones((2, 2)),
        w_mode_mn_np=np.ones((2, 2)),
        lambda_update_scale=1.0,
        lambda_update_scale_j=1.0,
        vmec2000_control=False,
        precond_cache=SimpleNamespace(valid=False, rz_norm=None, f_norm1=None),
        need_bcovar_update=True,
        host_update_assembly=False,
        use_fused_precond_output_scaling=False,
        adjoint_trace=False,
        adjoint_trace_mode="lite",
        accepted_control_ptau_arrays=None,
        ptau_pshalf_jax=None,
        ptau_ohs_jax=None,
        preconditioner_use_precomputed_tridi_policy=None,
        preconditioner_use_lax_tridi_policy=None,
        timing_enabled=True,
        timing_detail_enabled=True,
        timing_stats={},
        t_precond_start=0.0,
        perf_counter=lambda: 1.0,
        record_timing=lambda name, _start: timings.append(name),
        has_jax_func=lambda: False,
        block_until_ready=None,
        tomnsps_type=lambda **fields: SimpleNamespace(**fields),
        refresh_preconditioner_cache_func=lambda *_args, **_kwargs: None,
        scale_m1_precond_rhs_func=lambda frzl, _mats: frzl,
        rz_preconditioner_apply_func=lambda *_args, **_kwargs: None,
        rz_norm_func=lambda _state: 0.0,
        apply_vmec2000_preconditioner_runtime_func=lambda **_kwargs: None,
        radial_preconditioner_output_blocks_jax_func=lambda **_kwargs: _force_blocks(2.0),
        apply_radial_tridi_func=lambda x, _alpha: x,
        mode_weight_force_blocks_np_func=lambda blocks, **_kwargs: blocks._replace(frcc=blocks.frcc + 10.0),
        mode_weight_force_blocks_jax_func=lambda blocks, **_kwargs: blocks._replace(frcc=blocks.frcc + 20.0),
        zeros_coeff_np=np.zeros((2, 2)),
        rz_scale=1.0,
        l_scale=1.0,
        precond_radial_alpha=0.5,
        precond_lambda_alpha=0.5,
    )
    kwargs.update(overrides)
    return kwargs, timings


def test_radial_preconditioner_path_applies_mode_and_lambda_scaling() -> None:
    kwargs, timings = _common_kwargs(lambda_update_scale=3.0, lambda_update_scale_j=3.0)

    result = apply_residual_iteration_preconditioner(**kwargs)

    np.testing.assert_allclose(result.preconditioned_blocks.frcc, np.full((2, 2), 2.0))
    np.testing.assert_allclose(result.update_force_blocks.frcc, np.full((2, 2), 22.0))
    np.testing.assert_allclose(result.update_force_blocks.flsc, np.full((2, 2), (2.0 + 4.0) * 3.0))
    assert result.frzl_pre.flsc.shape == (2, 2)
    assert not result.outputs_scaled
    assert not result.fsq1_ready
    assert "precond_apply" in timings
    assert "precond_mode_scale" in timings
    assert "preconditioner" in timings


def test_scaled_vmec2000_preconditioner_path_preserves_fused_update_blocks() -> None:
    calls = {"mode": 0}
    update_blocks = _force_blocks(7.0)

    def fake_vmec2000_apply(**_kwargs):
        return SimpleNamespace(
            lam_prec=np.array([1.0]),
            mats={"dr": np.array([1.0])},
            jmax=3,
            cache_update_trace=True,
            blocks=_force_blocks(4.0),
            update_blocks=update_blocks,
            gcr2_p=1.0,
            gcz2_p=2.0,
            gcl2_p=3.0,
            fsqr1_safe=4.0,
            fsqz1_safe=5.0,
            fsql1_safe=6.0,
            fsq1_safe=15.0,
            frzl_rz=SimpleNamespace(frcc=np.array([1.0])),
            frzl_lam_pre=SimpleNamespace(flsc=np.array([2.0])),
            outputs_scaled=True,
            fsq1_ready=True,
            accepted_control_ptau_payload=("ptau",),
        )

    def unexpected_mode_scale(blocks, **_kwargs):
        calls["mode"] += 1
        return blocks

    kwargs, _timings = _common_kwargs(
        use_vmec2000_preconditioner=True,
        lambda_update_scale=9.0,
        lambda_update_scale_j=9.0,
        apply_vmec2000_preconditioner_runtime_func=fake_vmec2000_apply,
        mode_weight_force_blocks_jax_func=unexpected_mode_scale,
        mode_weight_force_blocks_np_func=unexpected_mode_scale,
    )

    result = apply_residual_iteration_preconditioner(**kwargs)

    assert result.update_force_blocks is update_blocks
    np.testing.assert_allclose(result.update_force_blocks.flsc, update_blocks.flsc)
    assert result.lam_prec.tolist() == [1.0]
    assert result.mats == {"dr": np.array([1.0])}
    assert result.jmax == 3
    assert result.cache_update_trace
    assert result.outputs_scaled
    assert result.fsq1_ready
    assert result.accepted_control_ptau_payload == ("ptau",)
    assert calls["mode"] == 0


def _scalar_common_kwargs(preconditioner_payload, *, frzl_pre=None, **overrides):
    timings: list[str] = []
    calls: dict[str, int] = {"host": 0, "jax": 0, "payload": 0, "dump": 0}

    def host_channels(**_kwargs):
        calls["host"] += 1
        return (10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 48.0)

    def jax_channels(**_kwargs):
        calls["jax"] += 1
        return (20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 78.0)

    def materialize_payload(**_kwargs):
        calls["payload"] += 1
        return SimpleNamespace(fsq1=99.0, accepted_control_ptau_host=(0.1, 0.2), control_payload_used=True)

    kwargs = dict(
        preconditioner_payload=preconditioner_payload,
        frzl_pre=frzl_pre if frzl_pre is not None else preconditioner_payload.frzl_pre,
        state=SimpleNamespace(),
        static=SimpleNamespace(cfg=SimpleNamespace(lconm1=True)),
        k=SimpleNamespace(),
        s=np.array([0.0, 1.0]),
        delta_s=0.5,
        host_update_assembly=False,
        host_fsq1_norms_on_accelerator=False,
        backend_name="cpu",
        vmec2000_control=False,
        precond_cache=SimpleNamespace(valid=False, rz_norm=None, f_norm1=None),
        need_bcovar_update=True,
        converged_physical=False,
        reference_mode=True,
        badjac_use_state=False,
        dump_ptau_state=False,
        dump_ptau_env="",
        timing_enabled=True,
        timing_stats={},
        t_fsq1_precond_norm_start=0.0,
        t_iteration_control_fsq1_start=0.0,
        perf_counter=lambda: 1.0,
        record_timing=lambda name, _start: timings.append(name),
        tree_has_tracer_func=lambda _x: False,
        tomnsps_to_numpy_host_func=lambda frzl: SimpleNamespace(host_copy=frzl),
        vmec_gcx2_from_tomnsps_np_func=lambda **_kwargs: (1.0, 2.0, 3.0),
        vmec_gcx2_from_tomnsps_func=lambda **_kwargs: (4.0, 5.0, 6.0),
        host_preconditioned_residual_scalar_channels_func=host_channels,
        jax_preconditioned_residual_scalar_channels_func=jax_channels,
        materialize_accepted_control_payload_func=materialize_payload,
        numpy_module=np,
        jnp_module=np,
        jax_module=SimpleNamespace(),
        rz_norm_np_func=lambda _state: 0.0,
        rz_norm_func=lambda _state: 0.0,
        lambda_preconditioned_full_norm_func=lambda *_args, **_kwargs: 0.0,
        finite_float_or_zero_func=lambda value: float(value),
        cached_or_current_f_norm1_jax_func=lambda **_kwargs: (0.0, 0.0),
        dump_lam_fsql1_func=lambda _fsql1: calls.__setitem__("dump", calls["dump"] + 1),
        device_get_floats_func=lambda *_args: (0.0, 0.0),
        accepted_control_ptau_host_from_payload_func=lambda *_args: (0.0, 0.0),
        scan_math_kernel_arrays_from_k_func=lambda _k: (),
        accepted_control_payload_jit_func=lambda *_args, **_kwargs: None,
        ptau_pshalf_jax=None,
        ptau_ohs_jax=None,
    )
    kwargs.update(overrides)
    return kwargs, timings, calls


def test_preconditioned_scalar_channels_host_path_skips_control_payload() -> None:
    preconditioned = apply_residual_iteration_preconditioner(**_common_kwargs()[0])
    kwargs, timings, calls = _scalar_common_kwargs(
        preconditioned,
        host_update_assembly=True,
        backend_name="cpu",
    )

    result = resolve_preconditioned_residual_scalars(**kwargs)

    assert result.use_host_fsq1_norms is False
    assert result.frzl_pre_host is preconditioned.frzl_pre
    assert result.gcr2_p == 1.0
    assert result.fsq1 == 48.0
    assert result.accepted_control_ptau_host is None
    assert calls == {"host": 1, "jax": 0, "payload": 0, "dump": 1}
    assert "iteration_control_fsq1_precond_norm" in timings
    assert "iteration_control_fsq1_scalar_build" not in timings
    assert "iteration_control_fsq1" in timings


def test_preconditioned_scalar_channels_device_path_materializes_control_payload() -> None:
    preconditioned = apply_residual_iteration_preconditioner(**_common_kwargs()[0])
    kwargs, timings, calls = _scalar_common_kwargs(
        preconditioned,
        backend_name="gpu",
        reference_mode=True,
        vmec2000_control=False,
    )

    result = resolve_preconditioned_residual_scalars(**kwargs)

    assert result.use_host_fsq1_norms is False
    assert result.frzl_pre_host is None
    assert result.gcr2_p == 4.0
    assert result.fsq1 == 99.0
    assert result.accepted_control_ptau_host == (0.1, 0.2)
    assert result.control_payload_used
    assert calls == {"host": 0, "jax": 1, "payload": 1, "dump": 1}
    assert "iteration_control_fsq1_precond_norm" in timings
    assert "iteration_control_fsq1_scalar_build" in timings
    assert "iteration_control_fsq1" in timings
