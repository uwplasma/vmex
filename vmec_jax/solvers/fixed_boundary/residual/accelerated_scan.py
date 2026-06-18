"""Accelerated fixed-boundary residual scan runner."""

from __future__ import annotations

from typing import Any

import numpy as np

from vmec_jax.solvers.fixed_boundary.results import SolveVmecResidualResult
from vmec_jax.state import VMECState


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
    scan_device_runtime_type: Any,
    scan_convergence_predicate_type: Any,
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
    scan_device_runtime = scan_device_runtime_type(
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
    ftol_j = jnp_module.asarray(float(ftol), dtype=dtype)
    fsq_total_target_j = None
    if fsq_total_target is not None:
        fsq_total_target_j = jnp_module.asarray(float(fsq_total_target), dtype=dtype)
    scan_converged = scan_convergence_predicate_type(
        ftol=ftol_j,
        fsq_total_target=fsq_total_target_j,
        converged_func=converged_residuals_func,
    )

    include_edge_scan = False
    compute_forces_scan = compute_forces if jit_forces else compute_forces_impl
    scan_cache_key = (
        "scan_v1",
        static_key,
        wout_key,
        edge_value_key,
        int(max_iter),
        float(step_size),
        float(initial_flip_sign),
        float(lambda_update_scale),
        float(precond_radial_alpha),
        float(precond_lambda_alpha),
        bool(apply_m1_constraints),
        bool(jit_forces),
    )
    if scan_timing_enabled and scan_total_start is not None:
        scan_timing_stats["scan_setup_s"] += perf_counter() - float(scan_total_start)

    def _scan_step(carry, it):
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
            frss_in = (frzl.frss if frzl.frss is not None else jnp_module.zeros_like(frzl.frcc)) * rz_scale[
                :, None, None
            ]
            fzcs_in = (frzl.fzcs if frzl.fzcs is not None else jnp_module.zeros_like(frzl.fzsc)) * rz_scale[
                :, None, None
            ]
            frcc, frss, fzsc, fzcs = apply_radial_tridi_batched(
                [
                    frzl.frcc * rz_scale[:, None, None],
                    frss_in,
                    frzl.fzsc * rz_scale[:, None, None],
                    fzcs_in,
                ],
                precond_radial_alpha,
            )
            flcs_in = (frzl.flcs if frzl.flcs is not None else jnp_module.zeros_like(frzl.flsc)) * l_scale[
                :, None, None
            ]
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
            frsc_u = (
                jnp_module.asarray(getattr(frzl, "frsc", None))
                if getattr(frzl, "frsc", None) is not None
                else jnp_module.zeros_like(frcc_u)
            ) * w_mode_mn[None, :, :]
            frcs_u = (
                jnp_module.asarray(getattr(frzl, "frcs", None))
                if getattr(frzl, "frcs", None) is not None
                else jnp_module.zeros_like(frcc_u)
            ) * w_mode_mn[None, :, :]
            fzcc_u = (
                jnp_module.asarray(getattr(frzl, "fzcc", None))
                if getattr(frzl, "fzcc", None) is not None
                else jnp_module.zeros_like(fzsc_u)
            ) * w_mode_mn[None, :, :]
            fzss_u = (
                jnp_module.asarray(getattr(frzl, "fzss", None))
                if getattr(frzl, "fzss", None) is not None
                else jnp_module.zeros_like(fzsc_u)
            ) * w_mode_mn[None, :, :]
            flcc_u = (
                jnp_module.asarray(getattr(frzl, "flcc", None))
                if getattr(frzl, "flcc", None) is not None
                else jnp_module.zeros_like(flsc_u)
            ) * w_mode_mn[None, :, :]
            flss_u = (
                jnp_module.asarray(getattr(frzl, "flss", None))
                if getattr(frzl, "flss", None) is not None
                else jnp_module.zeros_like(flsc_u)
            ) * w_mode_mn[None, :, :]

            if lambda_update_scale != 1.0:
                flsc_u = flsc_u * lambda_update_scale_j
                flcs_u = flcs_u * lambda_update_scale_j
                flcc_u = flcc_u * lambda_update_scale_j
                flss_u = flss_u * lambda_update_scale_j

            dR = (time_step_j * flip_sign_j) * mn_cos_to_signed_physical(frcc_u, frss_u)
            sin_updates = mode_context.mn_sin_to_signed_batch(
                jnp_module.stack([fzsc_u, flsc_u], axis=0),
                jnp_module.stack([fzcs_u, flcs_u], axis=0),
            )
            dZ = (time_step_j * flip_sign_j) * sin_updates[0]
            dL = (time_step_j * flip_sign_j) * sin_updates[1]
            if bool(cfg.lasym):
                dR_sin = (time_step_j * flip_sign_j) * mn_sin_to_signed_physical(frsc_u, frcs_u)
                dZ_cos = (time_step_j * flip_sign_j) * mn_cos_to_signed_physical(fzcc_u, fzss_u)
                dL_cos = (time_step_j * flip_sign_j) * mn_cos_to_signed_physical_lambda(flcc_u, flss_u)
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
            state_new = apply_vmec_lambda_axis_rules(state_new)
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

    def _run_scan(state_init):
        carry0 = (
            state_init,
            jnp_module.asarray(False),
            jnp_module.asarray(-1, dtype=jnp_module.int32),
            jnp_module.asarray(jnp_module.inf, dtype=dtype),
            jnp_module.asarray(jnp_module.inf, dtype=dtype),
            jnp_module.asarray(jnp_module.inf, dtype=dtype),
        )
        return jax_module.lax.scan(_scan_step, carry0, jnp_module.arange(max_iter, dtype=jnp_module.int32))

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
    carry_final, hist = run_scan(state)
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
