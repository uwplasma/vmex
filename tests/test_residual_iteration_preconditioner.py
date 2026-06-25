from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from vmec_jax.solvers.fixed_boundary.residual.iteration_preconditioner import (
    apply_residual_iteration_preconditioner,
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
