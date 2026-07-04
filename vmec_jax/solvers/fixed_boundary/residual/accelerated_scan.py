"""Accelerated fixed-boundary residual scan runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from vmec_jax.solvers.fixed_boundary.results import SolveVmecResidualResult
from vmec_jax.solvers.fixed_boundary.residual.scan_adapters import (
    ScanConvergencePredicate,
    ScanDeviceRuntime,
)
from vmec_jax.state import VMECState


@dataclass(frozen=True)
class AcceleratedScanWeightedBlocks:
    """Mode-weighted force blocks used by one accelerated scan update."""

    frcc: Any
    frss: Any
    fzsc: Any
    fzcs: Any
    flsc: Any
    flcs: Any
    frsc: Any
    frcs: Any
    fzcc: Any
    fzss: Any
    flcc: Any
    flss: Any


def _accelerated_scan_cache_key(
    *,
    static_key: Any,
    wout_key: Any,
    edge_value_key: Any,
    max_iter: int,
    has_fsq_total_target: bool,
    precond_radial_alpha: float,
    precond_lambda_alpha: float,
    apply_m1_constraints: bool,
    jit_forces: bool,
) -> tuple[Any, ...]:
    """Return the structural cache key for the compiled accelerated scan runner.

    Numerical scalar controls such as time step, flip sign, lambda scaling, and
    convergence tolerances are runtime operands of the compiled runner.  Keeping
    them out of this key lets same-shape solves reuse one executable instead of
    compiling again for routine optimizer or input-deck tolerance changes.
    """

    return (
        "scan_v2",
        static_key,
        wout_key,
        edge_value_key,
        int(max_iter),
        bool(has_fsq_total_target),
        float(precond_radial_alpha),
        float(precond_lambda_alpha),
        bool(apply_m1_constraints),
        bool(jit_forces),
    )


def _weighted_optional_scan_block(*, frzl: Any, name: str, like: Any, w_mode_mn: Any, jnp_module: Any) -> Any:
    value = getattr(frzl, name, None)
    base = jnp_module.asarray(value) if value is not None else jnp_module.zeros_like(like)
    return base * w_mode_mn[None, :, :]


def _accelerated_scan_weighted_blocks(
    *,
    frzl: Any,
    rz_scale: Any,
    l_scale: Any,
    w_mode_mn: Any,
    precond_radial_alpha: float,
    precond_lambda_alpha: float,
    lambda_update_scale_j: Any,
    apply_radial_tridi_batched: Any,
    jnp_module: Any,
) -> AcceleratedScanWeightedBlocks:
    """Precondition and mode-weight one residual payload inside the scan."""

    frss_in = (frzl.frss if frzl.frss is not None else jnp_module.zeros_like(frzl.frcc)) * rz_scale[:, None, None]
    fzcs_in = (frzl.fzcs if frzl.fzcs is not None else jnp_module.zeros_like(frzl.fzsc)) * rz_scale[:, None, None]
    frcc, frss, fzsc, fzcs = apply_radial_tridi_batched(
        [
            frzl.frcc * rz_scale[:, None, None],
            frss_in,
            frzl.fzsc * rz_scale[:, None, None],
            fzcs_in,
        ],
        precond_radial_alpha,
    )
    flcs_in = (frzl.flcs if frzl.flcs is not None else jnp_module.zeros_like(frzl.flsc)) * l_scale[:, None, None]
    flsc, flcs = apply_radial_tridi_batched(
        [
            frzl.flsc * l_scale[:, None, None],
            flcs_in,
        ],
        precond_lambda_alpha,
    )

    frcc_u = frcc * w_mode_mn[None, :, :]
    frss_u = frss * w_mode_mn[None, :, :]
    fzsc_u = fzsc * w_mode_mn[None, :, :]
    fzcs_u = fzcs * w_mode_mn[None, :, :]
    flsc_u = flsc * w_mode_mn[None, :, :]
    flcs_u = flcs * w_mode_mn[None, :, :]
    blocks = AcceleratedScanWeightedBlocks(
        frcc=frcc_u,
        frss=frss_u,
        fzsc=fzsc_u,
        fzcs=fzcs_u,
        flsc=flsc_u,
        flcs=flcs_u,
        frsc=_weighted_optional_scan_block(frzl=frzl, name="frsc", like=frcc_u, w_mode_mn=w_mode_mn, jnp_module=jnp_module),
        frcs=_weighted_optional_scan_block(frzl=frzl, name="frcs", like=frcc_u, w_mode_mn=w_mode_mn, jnp_module=jnp_module),
        fzcc=_weighted_optional_scan_block(frzl=frzl, name="fzcc", like=fzsc_u, w_mode_mn=w_mode_mn, jnp_module=jnp_module),
        fzss=_weighted_optional_scan_block(frzl=frzl, name="fzss", like=fzsc_u, w_mode_mn=w_mode_mn, jnp_module=jnp_module),
        flcc=_weighted_optional_scan_block(frzl=frzl, name="flcc", like=flsc_u, w_mode_mn=w_mode_mn, jnp_module=jnp_module),
        flss=_weighted_optional_scan_block(frzl=frzl, name="flss", like=flsc_u, w_mode_mn=w_mode_mn, jnp_module=jnp_module),
    )
    lambda_scale = jnp_module.asarray(lambda_update_scale_j, dtype=blocks.flsc.dtype)
    return AcceleratedScanWeightedBlocks(
        frcc=blocks.frcc,
        frss=blocks.frss,
        fzsc=blocks.fzsc,
        fzcs=blocks.fzcs,
        flsc=blocks.flsc * lambda_scale,
        flcs=blocks.flcs * lambda_scale,
        frsc=blocks.frsc,
        frcs=blocks.frcs,
        fzcc=blocks.fzcc,
        fzss=blocks.fzss,
        flcc=blocks.flcc * lambda_scale,
        flss=blocks.flss * lambda_scale,
    )


def _accelerated_scan_state_update(
    *,
    state_i: Any,
    static: Any,
    cfg: Any,
    blocks: AcceleratedScanWeightedBlocks,
    time_step_j: Any,
    flip_sign_j: Any,
    free_boundary_enabled: bool,
    edge_Rcos: Any,
    edge_Rsin: Any,
    edge_Zcos: Any,
    edge_Zsin: Any,
    idx00: int,
    mode_context: Any,
    mn_cos_to_signed_physical: Any,
    mn_sin_to_signed_physical: Any,
    mn_cos_to_signed_physical_lambda: Any,
    enforce_fixed_boundary_and_axis: Any,
    apply_vmec_lambda_axis_rules: Any,
    jnp_module: Any,
) -> VMECState:
    """Apply one accelerated scan update and re-enforce VMEC boundary rules."""

    dR = (time_step_j * flip_sign_j) * mn_cos_to_signed_physical(blocks.frcc, blocks.frss)
    sin_updates = mode_context.mn_sin_to_signed_batch(
        jnp_module.stack([blocks.fzsc, blocks.flsc], axis=0),
        jnp_module.stack([blocks.fzcs, blocks.flcs], axis=0),
    )
    dZ = (time_step_j * flip_sign_j) * sin_updates[0]
    dL = (time_step_j * flip_sign_j) * sin_updates[1]
    if bool(cfg.lasym):
        dR_sin = (time_step_j * flip_sign_j) * mn_sin_to_signed_physical(blocks.frsc, blocks.frcs)
        dZ_cos = (time_step_j * flip_sign_j) * mn_cos_to_signed_physical(blocks.fzcc, blocks.fzss)
        dL_cos = (time_step_j * flip_sign_j) * mn_cos_to_signed_physical_lambda(blocks.flcc, blocks.flss)
    else:
        dR_sin = jnp_module.zeros_like(dR)
        dZ_cos = jnp_module.zeros_like(dR)
        dL_cos = jnp_module.zeros_like(dR)

    state_new = VMECState(
        layout=state_i.layout,
        Rcos=jnp_module.asarray(state_i.Rcos) + dR,
        Rsin=jnp_module.asarray(state_i.Rsin) + dR_sin,
        Zcos=jnp_module.asarray(state_i.Zcos) + dZ_cos,
        Zsin=jnp_module.asarray(state_i.Zsin) + dZ,
        Lcos=jnp_module.asarray(state_i.Lcos) + dL_cos,
        Lsin=jnp_module.asarray(state_i.Lsin) + dL,
    )
    state_new = enforce_fixed_boundary_and_axis(
        state_new,
        static,
        edge_Rcos=edge_Rcos,
        edge_Rsin=edge_Rsin,
        edge_Zcos=edge_Zcos,
        edge_Zsin=edge_Zsin,
        enforce_edge=not bool(free_boundary_enabled),
        enforce_lambda_axis=True,
        idx00=idx00,
    )
    return apply_vmec_lambda_axis_rules(state_new)


def run_accelerated_residual_scan(
    *,
    state: Any,
    state0: Any,
    static: Any,
    cfg: Any,
    max_iter: int,
    step_size: float,
    initial_flip_sign: float,
    lambda_update_scale: float,
    lambda_update_scale_j: Any,
    ftol: float,
    fsq_total_target: float | None,
    precond_radial_alpha: float,
    precond_lambda_alpha: float,
    apply_m1_constraints: bool,
    jit_forces: bool,
    free_boundary_enabled: bool,
    static_key: Any,
    wout_key: Any,
    edge_value_key: Any,
    edge_Rcos: Any,
    edge_Rsin: Any,
    edge_Zcos: Any,
    edge_Zsin: Any,
    idx00: int,
    w_mode_mn: Any,
    mode_context: Any,
    compute_forces: Any,
    compute_forces_impl: Any,
    apply_radial_tridi_batched: Any,
    mn_cos_to_signed_physical: Any,
    mn_sin_to_signed_physical: Any,
    mn_cos_to_signed_physical_lambda: Any,
    enforce_fixed_boundary_and_axis: Any,
    apply_vmec_lambda_axis_rules: Any,
    attach_freeb_diag: Any,
    scan_timing_env: str,
    jax_module: Any,
    jnp_module: Any,
    jit_func: Any,
    scan_timing_enabled_func: Any,
    new_scan_timing_stats_func: Any,
    build_scan_timing_report_func: Any,
    converged_residuals_func: Any,
    scan_device_ready_recorder: Any,
    get_or_build_scan_runner_func: Any,
    scan_runner_cache: Any,
    jit_cache_get_func: Any,
    jit_cache_put_func: Any,
    record_scan_runner_cache_miss_categories_func: Any,
    perf_counter: Any,
    differentiating_scan: bool,
) -> SolveVmecResidualResult:
    """Run the non-VMEC2000 accelerated residual scan path.

    The calling solver injects JAX and cache hooks so legacy monkeypatch seams
    in ``residual.iteration`` keep working.
    """

    scan_timing_enabled = scan_timing_enabled_func(scan_timing_env)
    scan_timing_stats = new_scan_timing_stats_func()
    scan_total_start = perf_counter() if scan_timing_enabled else None
    scan_device_runtime = ScanDeviceRuntime(
        scan_timing_enabled=bool(scan_timing_enabled),
        stats=scan_timing_stats,
        perf_counter=perf_counter,
        block_until_ready=jax_module.block_until_ready,
        tree_map=jax_module.tree_util.tree_map,
        record_ready=scan_device_ready_recorder,
    )

    dtype = jnp_module.asarray(state0.Rcos).dtype
    time_step_j = jnp_module.asarray(float(step_size), dtype=dtype)
    flip_sign_j = jnp_module.asarray(float(initial_flip_sign), dtype=dtype)
    lambda_update_scale_j = jnp_module.asarray(lambda_update_scale_j, dtype=dtype)
    ftol_j = jnp_module.asarray(float(ftol), dtype=dtype)
    has_fsq_total_target = fsq_total_target is not None
    fsq_total_target_j = jnp_module.asarray(
        float(fsq_total_target) if has_fsq_total_target else float("inf"),
        dtype=dtype,
    )

    include_edge_scan = False
    compute_forces_scan = compute_forces if jit_forces else compute_forces_impl
    scan_cache_key = _accelerated_scan_cache_key(
        static_key=static_key,
        wout_key=wout_key,
        edge_value_key=edge_value_key,
        max_iter=int(max_iter),
        has_fsq_total_target=bool(has_fsq_total_target),
        precond_radial_alpha=float(precond_radial_alpha),
        precond_lambda_alpha=float(precond_lambda_alpha),
        apply_m1_constraints=bool(apply_m1_constraints),
        jit_forces=bool(jit_forces),
    )
    if scan_timing_enabled and scan_total_start is not None:
        scan_timing_stats["scan_setup_s"] += perf_counter() - float(scan_total_start)

    def _scan_step(carry, it_and_controls):
        it, time_step_dyn, flip_sign_dyn, lambda_update_scale_dyn, ftol_dyn, fsq_total_target_dyn = it_and_controls
        state_i, converged, converged_iter, last_fsqr, last_fsqz, last_fsql = carry
        it = jnp_module.asarray(it, dtype=jnp_module.int32)

        def _hold_step(_):
            return carry, (last_fsqr, last_fsqz, last_fsql)

        def _advance_step(_):
            iter_since_restart = it + 1
            zero_m1 = jnp_module.where(
                iter_since_restart < 2,
                jnp_module.asarray(1.0, dtype=dtype),
                jnp_module.asarray(0.0, dtype=dtype),
            )

            _k, frzl, fsqr, fsqz, fsql, rz_scale, l_scale, _norms = compute_forces_scan(
                state_i,
                include_edge=include_edge_scan,
                zero_m1=zero_m1,
                iter_idx=None,
            )
            blocks = _accelerated_scan_weighted_blocks(
                frzl=frzl,
                rz_scale=rz_scale,
                l_scale=l_scale,
                w_mode_mn=w_mode_mn,
                precond_radial_alpha=float(precond_radial_alpha),
                precond_lambda_alpha=float(precond_lambda_alpha),
                lambda_update_scale_j=lambda_update_scale_dyn,
                apply_radial_tridi_batched=apply_radial_tridi_batched,
                jnp_module=jnp_module,
            )
            state_new = _accelerated_scan_state_update(
                state_i=state_i,
                static=static,
                cfg=cfg,
                blocks=blocks,
                time_step_j=time_step_dyn,
                flip_sign_j=flip_sign_dyn,
                free_boundary_enabled=bool(free_boundary_enabled),
                edge_Rcos=edge_Rcos,
                edge_Rsin=edge_Rsin,
                edge_Zcos=edge_Zcos,
                edge_Zsin=edge_Zsin,
                idx00=int(idx00),
                mode_context=mode_context,
                mn_cos_to_signed_physical=mn_cos_to_signed_physical,
                mn_sin_to_signed_physical=mn_sin_to_signed_physical,
                mn_cos_to_signed_physical_lambda=mn_cos_to_signed_physical_lambda,
                enforce_fixed_boundary_and_axis=enforce_fixed_boundary_and_axis,
                apply_vmec_lambda_axis_rules=apply_vmec_lambda_axis_rules,
                jnp_module=jnp_module,
            )
            scan_converged = ScanConvergencePredicate(
                ftol=ftol_dyn,
                fsq_total_target=fsq_total_target_dyn if has_fsq_total_target else None,
                converged_func=converged_residuals_func,
            )
            conv_now = scan_converged(fsqr, fsqz, fsql)
            conv_iter_new = jnp_module.where(
                (converged_iter < 0) & conv_now,
                it + jnp_module.asarray(1, dtype=jnp_module.int32),
                converged_iter,
            )
            carry_new = (
                state_new,
                converged | conv_now,
                conv_iter_new,
                fsqr,
                fsqz,
                fsql,
            )
            return carry_new, (fsqr, fsqz, fsql)

        return jax_module.lax.cond(converged, _hold_step, _advance_step, operand=None)

    def _run_scan(
        state_init,
        time_step_dyn,
        flip_sign_dyn,
        lambda_update_scale_dyn,
        ftol_dyn,
        fsq_total_target_dyn,
    ):
        carry0 = (
            state_init,
            jnp_module.asarray(False),
            jnp_module.asarray(-1, dtype=jnp_module.int32),
            jnp_module.asarray(jnp_module.inf, dtype=dtype),
            jnp_module.asarray(jnp_module.inf, dtype=dtype),
            jnp_module.asarray(jnp_module.inf, dtype=dtype),
        )
        controls = (
            jnp_module.arange(max_iter, dtype=jnp_module.int32),
            jnp_module.broadcast_to(time_step_dyn, (max_iter,)),
            jnp_module.broadcast_to(flip_sign_dyn, (max_iter,)),
            jnp_module.broadcast_to(lambda_update_scale_dyn, (max_iter,)),
            jnp_module.broadcast_to(ftol_dyn, (max_iter,)),
            jnp_module.broadcast_to(fsq_total_target_dyn, (max_iter,)),
        )
        return jax_module.lax.scan(_scan_step, carry0, controls)

    scan_run_setup_start = perf_counter() if scan_timing_enabled else None
    run_scan, scan_runner_cache_status = get_or_build_scan_runner_func(
        _run_scan,
        cache=scan_runner_cache,
        key=scan_cache_key,
        differentiating_scan=bool(differentiating_scan),
        scan_timing_enabled=bool(scan_timing_enabled),
        scan_timing_stats=scan_timing_stats,
        jit_func=jit_func,
        cache_get=jit_cache_get_func,
        cache_put=jit_cache_put_func,
        record_miss_categories=record_scan_runner_cache_miss_categories_func,
        perf_counter=perf_counter,
    )
    if scan_timing_enabled and scan_run_setup_start is not None:
        scan_timing_stats["scan_run_setup_s"] += perf_counter() - float(scan_run_setup_start)

    scan_device_start = perf_counter() if scan_timing_enabled else None
    carry_final, hist = run_scan(
        state,
        time_step_j,
        flip_sign_j,
        lambda_update_scale_j,
        ftol_j,
        fsq_total_target_j,
    )
    if scan_timing_enabled and scan_device_start is not None:
        carry_final, hist = scan_device_runtime.ready(
            scan_device_start,
            (carry_final, hist),
            cache_status=scan_runner_cache_status,
        )
    scan_materialize_start = perf_counter() if scan_timing_enabled else None
    state_final, converged_final, converged_iter_final, _, _, _ = carry_final
    fsqr_hist, fsqz_hist, fsql_hist = hist
    w_hist = fsqr_hist + fsqz_hist + fsql_hist
    w_hist_np = np.asarray(w_hist)
    fsqr_hist_np = np.asarray(fsqr_hist)
    fsqz_hist_np = np.asarray(fsqz_hist)
    fsql_hist_np = np.asarray(fsql_hist)
    converged_host = bool(np.asarray(converged_final))
    converged_iter_host = int(np.asarray(converged_iter_final)) if converged_host else -1
    n_iter_host = int(converged_iter_host) if converged_host else int(max_iter)
    if scan_timing_enabled and scan_materialize_start is not None:
        scan_timing_stats["scan_host_materialize_s"] += perf_counter() - float(scan_materialize_start)
    scan_timing_report = None
    if scan_timing_enabled:
        scan_total_s = (
            perf_counter() - float(scan_total_start)
            if scan_total_start is not None
            else sum(scan_timing_stats.values())
        )
        scan_timing_report = build_scan_timing_report_func(
            iterations=int(n_iter_host),
            stats=scan_timing_stats,
            scan_total_s=float(scan_total_s),
        )
    res_scan_fast = SolveVmecResidualResult(
        state=state_final,
        n_iter=n_iter_host,
        w_history=w_hist_np,
        fsqr2_history=fsqr_hist_np,
        fsqz2_history=fsqz_hist_np,
        fsql2_history=fsql_hist_np,
        grad_rms_history=np.asarray([], dtype=float),
        step_history=np.asarray([], dtype=float),
        diagnostics={
            "use_scan": True,
            "accelerated_scan": True,
            "scan_path": "accelerated",
            "fsq_total_target": fsq_total_target,
            "converged": converged_host,
            "converged_iter": converged_iter_host,
            **({"timing": scan_timing_report} if scan_timing_report is not None else {}),
        },
    )
    return attach_freeb_diag(res_scan_fast)
