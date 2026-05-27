"""Fixed-boundary solvers.

The first solver milestone is a robust "inner solve" for the VMEC ``lambda`` field
with R/Z held fixed. This is useful for:

- validating the magnetic energy objective against VMEC2000 `wout` files,
- building toward a full fixed-boundary equilibrium solve.

Notes
-----
This module intentionally avoids optional dependencies (e.g. jaxopt). The current
implementation uses gradient descent with a simple backtracking line search.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from contextlib import nullcontext
import time
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, NamedTuple

import numpy as np

from ._compat import has_jax, jax, jnp, jit
from . import _solve_runtime
from . import solve_residual_iter_policy as _residual_iter_policy
from .solve_residual_iter_config import (
    HEAVY_DUMP_ENVS as _HEAVY_DUMP_ENVS,
    LIGHT_DUMP_ENVS as _LIGHT_DUMP_ENVS,
    bad_jacobian_tau_tolerance as _bad_jacobian_tau_tolerance,
    normalize_debug_print_mode as _normalize_debug_print_mode,  # noqa: F401 - re-exported for internal helpers/tests.
    parse_bad_jacobian_config as _parse_bad_jacobian_config,
    resolve_chunked_scan_config as _resolve_chunked_scan_config,
    resolve_debug_print_config as _resolve_debug_print_config,
    resolve_dump_history_config as _resolve_dump_history_config,
    resolve_nstep_screen as _resolve_nstep_screen,
    should_probe_bad_jacobian_state as _should_probe_bad_jacobian_state,
)
from .solve_residual_iter_policy import (
    append_residual_iter_history_record as _append_residual_iter_history_record,
    append_residual_iter_terminal_history as _append_residual_iter_terminal_history,
    host_restart_decision as _host_restart_decision,
    host_update_assembly_policy as _host_update_assembly_policy,
    resolve_light_history as _resolve_light_history,
    resolve_restart_flags as _resolve_restart_flags,
    residual_iter_history_record as _residual_iter_history_record,
    scan_fallback_decision as _scan_fallback_decision,
    vmec2000_scan_options_from_env as _vmec2000_scan_options_from_env,
    vmec2000_time_control_decision as _vmec2000_time_control_decision,
)
from .solve_residual_iter_runtime_helpers import (
    _build_residual_iter_timing_report,
    _build_resume_state_base,
    _converged_residuals_scan_fast as _runtime_converged_residuals_scan_fast,
    _format_residual_iter_timing_message,
    _maybe_dump_ptau as _runtime_maybe_dump_ptau,
    _maybe_print_nonscan_state_debug,
    _scan_block_until_ready,
    _scan_device_run_ready as _runtime_scan_device_run_ready,
    _scan_print_uses_debug_callback,
    _scan_print_uses_debug_print,
    _scan_print_uses_io_callback,
    _vmec_freeb_plascur_from_bcovar as _runtime_vmec_freeb_plascur_from_bcovar,
)
from .solve_residual_iter_update_helpers import (
    ResidualVelocityBlocks as _ResidualVelocityBlocks,
    host_momentum_update_np as _host_momentum_update_np,
)
from .field import TWOPI, b2_from_bsup, bsup_from_geom, bsup_from_sqrtg_lambda
from .fourier import eval_fourier_dtheta, eval_fourier_dzeta_phys
from .geom import eval_geom
from .grids import angle_steps
from .solve_diagnostics_io import (
    _dump_freeb_axis_trace_record,
    _dump_freeb_control_trace_record,
    _dump_time_control_trace_record,
    _finite_float_or_zero,
    _format_axis_coeff,  # noqa: F401 - re-exported for existing internal tests/importers.
    _format_checkpoint_log_row as _format_checkpoint_log_row,
    _format_evolve_trace_row as _format_evolve_trace_row,
    _format_freeb_control_trace_row as _format_freeb_control_trace_row,
    _format_time_control_log_row as _format_time_control_log_row,
    _format_time_control_trace_row,  # noqa: F401 - re-exported for existing internal tests/importers.
    _format_vmec2000_iter_row,  # noqa: F401 - re-exported for existing internal tests/importers.
    _legacy_dump_record_path as _legacy_dump_record_path,
    _legacy_single_dump_iter_selected as _legacy_single_dump_iter_selected,
    _materialize_adjoint_trace_array,
    _maybe_dump_checkpoint_record,
    _maybe_dump_evolve_trace_record,
    _maybe_dump_jacobian_terms_record,
    _maybe_dump_time_control_record,
    _normalize_adjoint_trace_mode,
    _normalize_resume_state_mode,
    _pack_resume_state_record,
    _should_print_vmec2000_row,
    _vmec2000_cadence_selected,
)
from .solve_options import (
    validate_fixed_boundary_gd_options,
    validate_fixed_boundary_lbfgs_options,
    validate_lambda_gd_options,
    validate_pressure_shape,
    validate_residual_gn_options,
    validate_residual_iteration_options,
    validate_residual_lbfgs_options,
)
from .solve_optimizer_helpers import (
    ensure_descent_direction as _ensure_descent_direction,
    lbfgs_curvature_tolerance as _resolve_lbfgs_curvature_tol,
    lbfgs_two_loop_direction as _lbfgs_two_loop_direction,
)
from .solve_profile_helpers import (
    _half_mesh_from_full_mesh,
    _icurv_full_mesh_from_indata,
    _mass_half_mesh_from_indata,
    _pressure_half_mesh_from_indata,
    _s_half_from_full_mesh_s,  # noqa: F401 - re-exported for existing internal tests/importers.
    _vmec_force_flux_profiles,
)
from .solve_residual_iter_geometry_helpers import (
    _m1_internal_to_physical_pair as _geometry_m1_internal_to_physical_pair,
    _mn_sin_to_signed_physical_batch as _geometry_mn_sin_to_signed_physical_batch,
    _rz_norm_np as _geometry_rz_norm_np,
)
from .solve_force_payload_helpers import (
    ForceBlocks as _ForceBlocks,
    normalize_force_blocks as _normalize_force_blocks,  # noqa: F401 - re-exported for internal tests/importers.
    preconditioner_output_blocks_np as _preconditioner_output_blocks_np,
    residual_force_payload_after_m1_scalxc as _residual_force_payload_after_m1_scalxc,
    zero_edge_rz_force_block as _zero_edge_rz_force_block,  # noqa: F401 - re-exported for internal tests/importers.
    zero_edge_rz_force_blocks as _zero_edge_rz_force_blocks,
)
from .solve_residual_objective_helpers import (
    assemble_residual_objective_terms as _assemble_residual_objective_terms,
    residual_objective_vector as _residual_objective_vector,
)
from .solve_scan_output import postprocess_vmec2000_scan_result, unpack_vmec2000_scan_histories
from .solve_scan_payload_helpers import (
    ScanStepFields as _ScanStepFields,
    current_scan_payload as _current_scan_payload,
    mask_scan_restart_force_payload as _mask_scan_restart_force_payload,  # noqa: F401 - re-exported for internal tests/importers.
    restart_scan_payload as _restart_scan_payload,
    select_scan_force_payload as _select_scan_force_payload,
    select_scan_step_fields as _select_scan_step_fields,
)
from .solve_scan_math_helpers import (
    _hold_step as _scan_math_hold_step,
    _kernel_arrays_from_k as _scan_math_kernel_arrays_from_k,
    _no_restart_updates as _scan_math_no_restart_updates,
    _ptau_minmax_from_k_host as _scan_math_ptau_minmax_from_k_host,
    _ptau_minmax_from_k_jax as _scan_math_ptau_minmax_from_k_jax,
    _restart_updates as _scan_math_restart_updates,
    _state_jacobian as _scan_math_state_jacobian,
)
from .solve_scan_debug_helpers import (
    _append_timecontrol_scan_trace_row,
    _emit_vmec2000_iter_row as _emit_scan_vmec2000_iter_row,
    _emit_scan_prints as _emit_scan_debug_prints,
    _print_axis_guess as _print_scan_axis_guess,
    _print_vmec2000_row as _print_scan_vmec2000_row,
    _record_scan_device_ready,
)
from .solve_scan_planning_helpers import (
    apply_state_only_scan_options as _apply_state_only_scan_options,
    build_scan_timing_report as _build_scan_timing_report,
    build_vmec2000_scan_cache_key as _build_vmec2000_scan_cache_key,
    new_scan_timing_stats as _new_scan_timing_stats,
    normalize_scan_print_mode as _normalize_scan_print_mode,
    resolve_scan_iteration_plan as _resolve_scan_iteration_plan,
    resolve_scan_preflight_iters as _resolve_scan_preflight_iters,
    resolve_scan_run_flags as _resolve_scan_run_flags,
    scan_chunk_settings as _resolve_scan_chunk_settings,
    scan_jit_forces_enabled as _scan_jit_forces_enabled,
    scan_jit_preflight_enabled as _scan_jit_preflight_enabled,
    scan_timing_enabled as _scan_timing_enabled,
    validate_vmec2000_scan_guards as _validate_vmec2000_scan_guards,
)
from .solve_scan_time_control import (
    scan_fallback_probe_update,
    scan_restart_decision,
    scan_restart_transition,
    scan_stage_spike_post_update,
    scan_time_control_scalars,
)
from .state import VMECState, pack_state, unpack_state
from .vmec_residue import vmec_gcx2_from_tomnsps
from .vmec_tomnsp import TomnspsRZL


_SCAN_RUNNER_CACHE: OrderedDict[tuple, Any] = OrderedDict()
_COMPUTE_FORCES_CACHE: OrderedDict[tuple, Any] = OrderedDict()
_STRICT_UPDATE_STEP_JIT_CACHE: OrderedDict[tuple, Any] = OrderedDict()
_PRECOND_OUTPUT_SCALE_JIT_CACHE: OrderedDict[tuple, Any] = OrderedDict()
_PRECOND_OUTPUT_PAYLOAD_JIT_CACHE: OrderedDict[tuple, Any] = OrderedDict()
_PRECOND_APPLY_PAYLOAD_JIT_CACHE: OrderedDict[tuple, Any] = OrderedDict()
_ACCEPTED_CONTROL_PAYLOAD_JIT_CACHE: OrderedDict[tuple, Any] = OrderedDict()


_HostRestartDecision = _residual_iter_policy.HostRestartDecision
_ResidualIterHistoryRecord = _residual_iter_policy.ResidualIterHistoryRecord
_Vmec2000ScanOptions = _residual_iter_policy.Vmec2000ScanOptions
_Vmec2000TimeControlDecision = _residual_iter_policy.Vmec2000TimeControlDecision

_m1_internal_to_physical_pair = _geometry_m1_internal_to_physical_pair
_mn_sin_to_signed_physical_batch = _geometry_mn_sin_to_signed_physical_batch
_rz_norm_np = _geometry_rz_norm_np


def _resolve_preconditioner_tridi_policies(
    *, use_precomputed: bool | None
) -> tuple[bool, bool]:
    env_precomputed = os.getenv("VMEC_JAX_TRIDI_PRECOMPUTE", "0").strip().lower() not in (
        "",
        "0",
        "false",
        "no",
    )
    env_lax_tridi = os.getenv("VMEC_JAX_TRIDI_SOLVE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "lax",
        "force",
    )
    return (
        bool(env_precomputed) if use_precomputed is None else bool(use_precomputed),
        bool(env_lax_tridi),
    )


def _jit_cache_limit(env_name: str, default: int) -> int:
    """Return a non-negative JIT-cache size limit from an environment variable."""

    raw = os.getenv(env_name, str(default)).strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return max(0, int(default))


def _jit_cache_get(cache: OrderedDict[tuple, Any], key: tuple):
    cached = cache.get(key)
    if cached is not None:
        cache.move_to_end(key)
    return cached


def _jit_cache_put(cache: OrderedDict[tuple, Any], key: tuple, value, *, env_name: str, default: int):
    limit = _jit_cache_limit(env_name, default)
    if limit == 0:
        return value
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > limit:
        cache.popitem(last=False)
    return value


def _strict_update_static_cache_key(static) -> tuple[Any, ...]:
    """Return a structural cache key for strict-update kernels.

    The strict-update kernel only depends on VMEC mode/radial layout metadata,
    not on the Python object identity of ``static``.  Keying on ``id(static)``
    caused accepted-point exact optimizer solves to compile/cache a new GPU
    update kernel for each otherwise-identical callback.
    """

    cfg = static.cfg
    modes = getattr(static, "modes", None)
    s = getattr(static, "s", None)
    s_shape = tuple(getattr(s, "shape", ()))
    s_dtype = str(getattr(s, "dtype", ""))
    return (
        int(getattr(cfg, "ns", 0)),
        int(getattr(cfg, "mpol", 0)),
        int(getattr(cfg, "ntor", 0)),
        int(getattr(cfg, "nfp", 0)),
        bool(getattr(cfg, "lasym", False)),
        bool(getattr(cfg, "lthreed", False)),
        int(getattr(modes, "K", 0)) if modes is not None else 0,
        s_shape,
        s_dtype,
    )


def _strict_update_step_jit(
    static,
    *,
    limit_update_rms: bool,
    need_update_rms: bool,
    divide_by_scalxc_for_update: bool,
    enforce_edge: bool = True,
):
    """Return a cached fused strict-update step for accelerator exact solves."""
    if not has_jax():
        return None
    key = (
        _strict_update_static_cache_key(static),
        bool(limit_update_rms),
        bool(need_update_rms),
        bool(divide_by_scalxc_for_update),
        bool(enforce_edge),
    )
    cached = _jit_cache_get(_STRICT_UPDATE_STEP_JIT_CACHE, key)
    if cached is not None:
        return cached

    from .discrete_adjoint import strict_update_accepted_step

    def _step(
        state_pre,
        dt_eff,
        b1,
        fac,
        force_scale,
        flip_sign,
        vRcc_before,
        vRss_before,
        vZsc_before,
        vZcs_before,
        vLsc_before,
        vLcs_before,
        vRsc_before,
        vRcs_before,
        vZcc_before,
        vZss_before,
        vLcc_before,
        vLss_before,
        frcc_u,
        frss_u,
        fzsc_u,
        fzcs_u,
        flsc_u,
        flcs_u,
        frsc_u,
        frcs_u,
        fzcc_u,
        fzss_u,
        flcc_u,
        flss_u,
        max_update_rms,
    ):
        return strict_update_accepted_step(
            state_pre,
            static,
            dt_eff=dt_eff,
            b1=b1,
            fac=fac,
            force_scale=force_scale,
            flip_sign=flip_sign,
            vRcc_before=vRcc_before,
            vRss_before=vRss_before,
            vZsc_before=vZsc_before,
            vZcs_before=vZcs_before,
            vLsc_before=vLsc_before,
            vLcs_before=vLcs_before,
            vRsc_before=vRsc_before,
            vRcs_before=vRcs_before,
            vZcc_before=vZcc_before,
            vZss_before=vZss_before,
            vLcc_before=vLcc_before,
            vLss_before=vLss_before,
            frcc_u=frcc_u,
            frss_u=frss_u,
            fzsc_u=fzsc_u,
            fzcs_u=fzcs_u,
            flsc_u=flsc_u,
            flcs_u=flcs_u,
            frsc_u=frsc_u,
            frcs_u=frcs_u,
            fzcc_u=fzcc_u,
            fzss_u=fzss_u,
            flcc_u=flcc_u,
            flss_u=flss_u,
            max_update_rms=max_update_rms,
            limit_update_rms=bool(limit_update_rms),
            need_update_rms=bool(need_update_rms),
            divide_by_scalxc_for_update=bool(divide_by_scalxc_for_update),
            enforce_edge=bool(enforce_edge),
        )

    compiled = jax.jit(_step)
    return _jit_cache_put(
        _STRICT_UPDATE_STEP_JIT_CACHE,
        key,
        compiled,
        env_name="VMEC_JAX_STRICT_UPDATE_CACHE_SIZE",
        default=16,
    )


def _preconditioner_output_scaling_jit(*, apply_lambda_update_scale: bool):
    """Return a cached fused scaler for R/Z/lambda preconditioner outputs."""
    if not has_jax():
        return None
    key = (bool(apply_lambda_update_scale),)
    cached = _jit_cache_get(_PRECOND_OUTPUT_SCALE_JIT_CACHE, key)
    if cached is not None:
        return cached

    def _scale(frzl_rz, lam_prec, w_mode_mn, lambda_update_scale_j):
        w = jnp.asarray(w_mode_mn)[None, :, :]
        lam_prec_j = jnp.asarray(lam_prec)

        frcc = jnp.asarray(frzl_rz.frcc)
        frss = frzl_rz.frss
        fzsc = jnp.asarray(frzl_rz.fzsc)
        fzcs = frzl_rz.fzcs

        flsc = jnp.asarray(frzl_rz.flsc) * lam_prec_j
        flcs = None if frzl_rz.flcs is None else (jnp.asarray(frzl_rz.flcs) * lam_prec_j)

        frsc = (
            jnp.asarray(frzl_rz.frsc)
            if getattr(frzl_rz, "frsc", None) is not None
            else jnp.zeros_like(frcc)
        )
        frcs = (
            jnp.asarray(frzl_rz.frcs)
            if getattr(frzl_rz, "frcs", None) is not None
            else jnp.zeros_like(frcc)
        )
        fzcc = (
            jnp.asarray(frzl_rz.fzcc)
            if getattr(frzl_rz, "fzcc", None) is not None
            else jnp.zeros_like(fzsc)
        )
        fzss = (
            jnp.asarray(frzl_rz.fzss)
            if getattr(frzl_rz, "fzss", None) is not None
            else jnp.zeros_like(fzsc)
        )
        flcc = (
            jnp.asarray(frzl_rz.flcc) * lam_prec_j
            if getattr(frzl_rz, "flcc", None) is not None
            else jnp.zeros_like(flsc)
        )
        flss = (
            jnp.asarray(frzl_rz.flss) * lam_prec_j
            if getattr(frzl_rz, "flss", None) is not None
            else jnp.zeros_like(flsc)
        )

        frcc_u = frcc * w
        frss_u = (frss if frss is not None else jnp.zeros_like(frcc_u)) * w
        fzsc_u = fzsc * w
        fzcs_u = (fzcs if fzcs is not None else jnp.zeros_like(fzsc_u)) * w
        flsc_u = flsc * w
        flcs_u = (flcs if flcs is not None else jnp.zeros_like(flsc_u)) * w
        frsc_u = frsc * w
        frcs_u = frcs * w
        fzcc_u = fzcc * w
        fzss_u = fzss * w
        flcc_u = flcc * w
        flss_u = flss * w

        if bool(apply_lambda_update_scale):
            lambda_update_scale_j = jnp.asarray(lambda_update_scale_j, dtype=flsc_u.dtype)
            flsc_u = flsc_u * lambda_update_scale_j
            flcs_u = flcs_u * lambda_update_scale_j
            flcc_u = flcc_u * lambda_update_scale_j
            flss_u = flss_u * lambda_update_scale_j

        return (
            (frcc, frss, fzsc, fzcs, flsc, flcs, frsc, frcs, fzcc, fzss, flcc, flss),
            (frcc_u, frss_u, fzsc_u, fzcs_u, flsc_u, flcs_u, frsc_u, frcs_u, fzcc_u, fzss_u, flcc_u, flss_u),
        )

    compiled = jax.jit(_scale)
    return _jit_cache_put(
        _PRECOND_OUTPUT_SCALE_JIT_CACHE,
        key,
        compiled,
        env_name="VMEC_JAX_PRECOND_OUTPUT_SCALE_CACHE_SIZE",
        default=4,
    )


def _preconditioner_output_payload_jit(
    *,
    apply_lambda_update_scale: bool,
    vmec2000_control: bool,
    lconm1: bool,
):
    """Return a cached GPU payload builder for preconditioner outputs and fsq1.

    This keeps the VMEC2000 convention that ``lambda_update_scale`` only changes
    coefficient updates; the preconditioned lambda residual diagnostic still uses
    the unscaled ``faclam*gcl`` norm.
    """

    if not has_jax():
        return None
    key = (bool(apply_lambda_update_scale), bool(vmec2000_control), bool(lconm1))
    cached = _jit_cache_get(_PRECOND_OUTPUT_PAYLOAD_JIT_CACHE, key)
    if cached is not None:
        return cached

    def _payload(frzl_rz, lam_prec, w_mode_mn, lambda_update_scale_j, f_norm1, delta_s, s):
        (pre_blocks, update_blocks) = _preconditioner_output_scaling_jit(
            apply_lambda_update_scale=bool(apply_lambda_update_scale)
        )(frzl_rz, lam_prec, w_mode_mn, lambda_update_scale_j)
        (frcc, frss, fzsc, fzcs, flsc, flcs, frsc, frcs, fzcc, fzss, flcc, flss) = pre_blocks
        frzl_pre = TomnspsRZL(
            frcc=frcc,
            frss=frss,
            fzsc=fzsc,
            fzcs=fzcs,
            flsc=flsc,
            flcs=flcs,
            frsc=frsc,
            frcs=frcs,
            fzcc=fzcc,
            fzss=fzss,
            flcc=flcc,
            flss=flss,
        )
        gcr2_p, gcz2_p, gcl2_p = vmec_gcx2_from_tomnsps(
            frzl=frzl_pre,
            lconm1=bool(lconm1),
            apply_m1_constraints=False,
            include_edge=True,
            apply_scalxc=False,
            s=s,
        )
        f_norm1_j = jnp.asarray(f_norm1)
        finite_fnorm1 = jnp.isfinite(f_norm1_j)
        fsqr1 = jnp.where(finite_fnorm1, gcr2_p * f_norm1_j, jnp.asarray(0.0, dtype=jnp.asarray(gcr2_p).dtype))
        fsqz1 = jnp.where(finite_fnorm1, gcz2_p * f_norm1_j, jnp.asarray(0.0, dtype=jnp.asarray(gcz2_p).dtype))
        if bool(vmec2000_control):
            gcl2_full = _lambda_preconditioned_full_norm(frzl_pre, use_jax=True)
            fsql1 = gcl2_full * delta_s
        else:
            fsql1 = gcl2_p * delta_s
        fsqr1_safe = jnp.where(jnp.isfinite(fsqr1), fsqr1, jnp.asarray(0.0, dtype=jnp.asarray(fsqr1).dtype))
        fsqz1_safe = jnp.where(jnp.isfinite(fsqz1), fsqz1, jnp.asarray(0.0, dtype=jnp.asarray(fsqz1).dtype))
        fsql1_safe = jnp.where(jnp.isfinite(fsql1), fsql1, jnp.asarray(0.0, dtype=jnp.asarray(fsql1).dtype))
        fsq1_safe = fsqr1_safe + fsqz1_safe + fsql1_safe
        return (
            pre_blocks,
            update_blocks,
            (gcr2_p, gcz2_p, gcl2_p, fsqr1_safe, fsqz1_safe, fsql1_safe, fsq1_safe),
        )

    compiled = jax.jit(_payload)
    return _jit_cache_put(
        _PRECOND_OUTPUT_PAYLOAD_JIT_CACHE,
        key,
        compiled,
        env_name="VMEC_JAX_PRECOND_OUTPUT_PAYLOAD_CACHE_SIZE",
        default=4,
    )


def _preconditioner_apply_payload_jit(
    *,
    jmax: int,
    lthreed: bool,
    lasym: bool,
    use_precomputed: bool,
    use_lax_tridi: bool,
    has_lax_t: bool,
    has_frss: bool,
    has_fzcs: bool,
    has_frsc: bool,
    has_frcs: bool,
    has_fzcc: bool,
    has_fzss: bool,
    has_flcs: bool,
    has_flcc: bool,
    has_flss: bool,
    apply_lambda_update_scale: bool,
    vmec2000_control: bool,
    lconm1: bool,
    include_control_ptau: bool,
):
    """Return a cached fused GPU preconditioner-apply/payload kernel.

    The accelerator exact path previously dispatched one kernel for the R/Z
    tridiagonal preconditioner apply and another for force-channel scaling plus
    preconditioned residual diagnostics.  Fusing those two hot kernels reduces
    accepted-step launch overhead while preserving the CPU host path.
    """

    if not has_jax():
        return None
    key = (
        int(jmax),
        bool(lthreed),
        bool(lasym),
        bool(use_precomputed),
        bool(use_lax_tridi and has_lax_t),
        bool(has_lax_t),
        bool(has_frss),
        bool(has_fzcs),
        bool(has_frsc),
        bool(has_frcs),
        bool(has_fzcc),
        bool(has_fzss),
        bool(has_flcs),
        bool(has_flcc),
        bool(has_flss),
        bool(apply_lambda_update_scale),
        bool(vmec2000_control),
        bool(lconm1),
        bool(include_control_ptau),
    )
    cached = _jit_cache_get(_PRECOND_APPLY_PAYLOAD_JIT_CACHE, key)
    if cached is not None:
        return cached

    from .preconditioner_1d_jax import _rz_preconditioner_apply_arrays

    use_rss = bool(lthreed)
    use_rsc = bool(lasym)
    use_rcs = bool(lthreed and lasym)
    use_zcs = bool(lthreed)
    use_zcc = bool(lasym)
    use_zss = bool(lthreed and lasym)

    def _payload_from_rz(
        frcc_rz,
        frss_rz,
        fzsc_rz,
        fzcs_rz,
        frsc_rz,
        frcs_rz,
        fzcc_rz,
        fzss_rz,
        flsc,
        flcs,
        flcc,
        flss,
        lam_prec,
        w_mode_mn,
        lambda_update_scale_j,
        f_norm1,
        delta_s,
        s,
        control_args,
    ):
        w = jnp.asarray(w_mode_mn)[None, :, :]
        lam_prec_j = jnp.asarray(lam_prec)

        frcc = jnp.asarray(frcc_rz)
        frss = jnp.asarray(frss_rz) if bool(has_frss) else None
        fzsc = jnp.asarray(fzsc_rz)
        fzcs = jnp.asarray(fzcs_rz) if bool(has_fzcs) else None
        flsc_pre = jnp.asarray(flsc) * lam_prec_j
        flcs_pre = (jnp.asarray(flcs) * lam_prec_j) if bool(has_flcs) else None

        frsc = jnp.asarray(frsc_rz) if bool(has_frsc) else jnp.zeros_like(frcc)
        frcs = jnp.asarray(frcs_rz) if bool(has_frcs) else jnp.zeros_like(frcc)
        fzcc = jnp.asarray(fzcc_rz) if bool(has_fzcc) else jnp.zeros_like(fzsc)
        fzss = jnp.asarray(fzss_rz) if bool(has_fzss) else jnp.zeros_like(fzsc)
        flcc_pre = (jnp.asarray(flcc) * lam_prec_j) if bool(has_flcc) else jnp.zeros_like(flsc_pre)
        flss_pre = (jnp.asarray(flss) * lam_prec_j) if bool(has_flss) else jnp.zeros_like(flsc_pre)

        frcc_u = frcc * w
        frss_u = (frss if frss is not None else jnp.zeros_like(frcc_u)) * w
        fzsc_u = fzsc * w
        fzcs_u = (fzcs if fzcs is not None else jnp.zeros_like(fzsc_u)) * w
        flsc_u = flsc_pre * w
        flcs_u = (flcs_pre if flcs_pre is not None else jnp.zeros_like(flsc_u)) * w
        frsc_u = frsc * w
        frcs_u = frcs * w
        fzcc_u = fzcc * w
        fzss_u = fzss * w
        flcc_u = flcc_pre * w
        flss_u = flss_pre * w

        if bool(apply_lambda_update_scale):
            lambda_update_scale_j = jnp.asarray(lambda_update_scale_j, dtype=flsc_u.dtype)
            flsc_u = flsc_u * lambda_update_scale_j
            flcs_u = flcs_u * lambda_update_scale_j
            flcc_u = flcc_u * lambda_update_scale_j
            flss_u = flss_u * lambda_update_scale_j

        frzl_pre = TomnspsRZL(
            frcc=frcc,
            frss=frss,
            fzsc=fzsc,
            fzcs=fzcs,
            flsc=flsc_pre,
            flcs=flcs_pre,
            frsc=frsc,
            frcs=frcs,
            fzcc=fzcc,
            fzss=fzss,
            flcc=flcc_pre,
            flss=flss_pre,
        )
        gcr2_p, gcz2_p, gcl2_p = vmec_gcx2_from_tomnsps(
            frzl=frzl_pre,
            lconm1=bool(lconm1),
            apply_m1_constraints=False,
            include_edge=True,
            apply_scalxc=False,
            s=s,
        )
        f_norm1_j = jnp.asarray(f_norm1)
        finite_fnorm1 = jnp.isfinite(f_norm1_j)
        fsqr1 = jnp.where(finite_fnorm1, gcr2_p * f_norm1_j, jnp.asarray(0.0, dtype=jnp.asarray(gcr2_p).dtype))
        fsqz1 = jnp.where(finite_fnorm1, gcz2_p * f_norm1_j, jnp.asarray(0.0, dtype=jnp.asarray(gcz2_p).dtype))
        if bool(vmec2000_control):
            gcl2_full = _lambda_preconditioned_full_norm(frzl_pre, use_jax=True)
            fsql1 = gcl2_full * delta_s
        else:
            fsql1 = gcl2_p * delta_s
        fsqr1_safe = jnp.where(jnp.isfinite(fsqr1), fsqr1, jnp.asarray(0.0, dtype=jnp.asarray(fsqr1).dtype))
        fsqz1_safe = jnp.where(jnp.isfinite(fsqz1), fsqz1, jnp.asarray(0.0, dtype=jnp.asarray(fsqz1).dtype))
        fsql1_safe = jnp.where(jnp.isfinite(fsql1), fsql1, jnp.asarray(0.0, dtype=jnp.asarray(fsql1).dtype))
        fsq1_safe = fsqr1_safe + fsqz1_safe + fsql1_safe
        if bool(include_control_ptau):
            (
                pru_even,
                pru_odd,
                pzu_even,
                pzu_odd,
                pr1_even,
                pr1_odd,
                pz1_even,
                pz1_odd,
                pshalf,
                ohs,
            ) = control_args
            ptau_min, ptau_max = _ptau_compute_jit(
                pru_even,
                pru_odd,
                pzu_even,
                pzu_odd,
                pr1_even,
                pr1_odd,
                pz1_even,
                pz1_odd,
                pshalf,
                ohs,
            )
            return (
                (frcc, frss, fzsc, fzcs, flsc_pre, flcs_pre, frsc, frcs, fzcc, fzss, flcc_pre, flss_pre),
                (frcc_u, frss_u, fzsc_u, fzcs_u, flsc_u, flcs_u, frsc_u, frcs_u, fzcc_u, fzss_u, flcc_u, flss_u),
                (gcr2_p, gcz2_p, gcl2_p, fsqr1_safe, fsqz1_safe, fsql1_safe, fsq1_safe),
                (fsq1_safe, ptau_min, ptau_max),
            )
        return (
            (frcc, frss, fzsc, fzcs, flsc_pre, flcs_pre, frsc, frcs, fzcc, fzss, flcc_pre, flss_pre),
            (frcc_u, frss_u, fzsc_u, fzcs_u, flsc_u, flcs_u, frsc_u, frcs_u, fzcc_u, fzss_u, flcc_u, flss_u),
            (gcr2_p, gcz2_p, gcl2_p, fsqr1_safe, fsqz1_safe, fsql1_safe, fsq1_safe),
        )

    @jit
    def _apply_payload(
        frcc,
        fzsc,
        frss,
        fzcs,
        frsc,
        frcs,
        fzcc,
        fzss,
        ar,
        br,
        dr,
        cr,
        ir,
        az,
        bz,
        dz,
        cz,
        iz,
        dlr_t,
        dr_t,
        dur_t,
        dlz_t,
        dz_t,
        duz_t,
        flsc,
        flcs,
        flcc,
        flss,
        lam_prec,
        w_mode_mn,
        lambda_update_scale_j,
        f_norm1,
        delta_s,
        s,
        *control_args,
    ):
        frcc_rz, frss_rz, fzsc_rz, fzcs_rz, frsc_rz, frcs_rz, fzcc_rz, fzss_rz = (
            _rz_preconditioner_apply_arrays(
                ar=ar,
                br=br,
                dr=dr,
                cr=cr,
                ir=ir,
                az=az,
                bz=bz,
                dz=dz,
                cz=cz,
                iz=iz,
                dlr_t=dlr_t,
                dr_t=dr_t,
                dur_t=dur_t,
                dlz_t=dlz_t,
                dz_t=dz_t,
                duz_t=duz_t,
                frcc=frcc,
                frss=frss,
                fzsc=fzsc,
                fzcs=fzcs,
                frsc=frsc,
                frcs=frcs,
                fzcc=fzcc,
                fzss=fzss,
                jmax=int(jmax),
                use_precomputed=bool(use_precomputed),
                use_lax_tridi=bool(use_lax_tridi and has_lax_t),
                use_rss=use_rss,
                use_rsc=use_rsc,
                use_rcs=use_rcs,
                use_zcs=use_zcs,
                use_zcc=use_zcc,
                use_zss=use_zss,
            )
        )
        return _payload_from_rz(
            frcc_rz,
            frss_rz,
            fzsc_rz,
            fzcs_rz,
            frsc_rz,
            frcs_rz,
            fzcc_rz,
            fzss_rz,
            flsc,
            flcs,
            flcc,
            flss,
            lam_prec,
            w_mode_mn,
            lambda_update_scale_j,
            f_norm1,
            delta_s,
            s,
            control_args,
        )

    return _jit_cache_put(
        _PRECOND_APPLY_PAYLOAD_JIT_CACHE,
        key,
        _apply_payload,
        env_name="VMEC_JAX_PRECOND_APPLY_PAYLOAD_CACHE_SIZE",
        default=8,
    )


def _accepted_control_payload_jit():
    """Return a cached JIT helper for accepted-step scalar control payloads.

    The non-scan VMEC control loop needs host scalars for Python time-step,
    restart, and Jacobian-sign decisions. On accelerators, pulling ``fsq1`` and
    the ``ptau`` extrema in separate synchronizations leaves tiny kernels on the
    critical path. This helper keeps those scalar reductions in one JAX payload
    and lets the caller materialize the tuple with a single ``device_get``.
    """

    if not has_jax():
        return None
    key: tuple[Any, ...] = ()
    cached = _jit_cache_get(_ACCEPTED_CONTROL_PAYLOAD_JIT_CACHE, key)
    if cached is not None:
        return cached

    @jax.jit
    def _payload(
        fsq1_safe,
        pru_even,
        pru_odd,
        pzu_even,
        pzu_odd,
        pr1_even,
        pr1_odd,
        pz1_even,
        pz1_odd,
        pshalf,
        ohs,
    ):
        ptau_min, ptau_max = _ptau_compute_jit(
            pru_even,
            pru_odd,
            pzu_even,
            pzu_odd,
            pr1_even,
            pr1_odd,
            pz1_even,
            pz1_odd,
            pshalf,
            ohs,
        )
        return jnp.asarray(fsq1_safe), ptau_min, ptau_max

    return _jit_cache_put(
        _ACCEPTED_CONTROL_PAYLOAD_JIT_CACHE,
        key,
        _payload,
        env_name="VMEC_JAX_ACCEPTED_CONTROL_PAYLOAD_CACHE_SIZE",
        default=2,
    )


def _preconditioner_apply_payload_fused(
    *,
    frzl_in: TomnspsRZL,
    mats: dict[str, Any],
    jmax: int,
    cfg,
    lam_prec,
    w_mode_mn,
    lambda_update_scale_j,
    f_norm1,
    delta_s,
    s,
    use_precomputed: bool | None,
    use_lax_tridi: bool | None,
    apply_lambda_update_scale: bool,
    vmec2000_control: bool,
    lconm1: bool,
    include_control_ptau: bool = False,
    control_ptau_arrays: tuple[Any, ...] | None = None,
    control_ptau_pshalf: Any = None,
    control_ptau_ohs: Any = None,
):
    """Apply R/Z preconditioning and build update/diagnostic payload in one dispatch."""

    from .preconditioner_1d_jax import _get_env_tridi_flags

    lthreed = bool(getattr(cfg, "lthreed", False))
    lasym = bool(getattr(cfg, "lasym", False))
    if use_precomputed is None or use_lax_tridi is None:
        env_pre, env_lax = _get_env_tridi_flags()
        if use_precomputed is None:
            use_precomputed = env_pre
        if use_lax_tridi is None:
            use_lax_tridi = env_lax
    has_cr_ir = ("cr" in mats) and ("ir" in mats) and ("cz" in mats) and ("iz" in mats)
    if not has_cr_ir:
        use_precomputed = False

    has_frss = frzl_in.frss is not None
    has_fzcs = frzl_in.fzcs is not None
    has_frsc = getattr(frzl_in, "frsc", None) is not None
    has_frcs = getattr(frzl_in, "frcs", None) is not None
    has_fzcc = getattr(frzl_in, "fzcc", None) is not None
    has_fzss = getattr(frzl_in, "fzss", None) is not None
    has_flcs = frzl_in.flcs is not None
    has_flcc = getattr(frzl_in, "flcc", None) is not None
    has_flss = getattr(frzl_in, "flss", None) is not None
    include_control_ptau = (
        bool(include_control_ptau)
        and control_ptau_arrays is not None
        and len(tuple(control_ptau_arrays)) == 8
    )

    has_lax_t = (
        ("dlr_t" in mats)
        and ("dr_t" in mats)
        and ("dur_t" in mats)
        and ("dlz_t" in mats)
        and ("dz_t" in mats)
        and ("duz_t" in mats)
    )
    apply_payload = _preconditioner_apply_payload_jit(
        jmax=int(jmax),
        lthreed=lthreed,
        lasym=lasym,
        use_precomputed=bool(use_precomputed),
        use_lax_tridi=bool(use_lax_tridi),
        has_lax_t=has_lax_t,
        has_frss=has_frss,
        has_fzcs=has_fzcs,
        has_frsc=has_frsc,
        has_frcs=has_frcs,
        has_fzcc=has_fzcc,
        has_fzss=has_fzss,
        has_flcs=has_flcs,
        has_flcc=has_flcc,
        has_flss=has_flss,
        apply_lambda_update_scale=bool(apply_lambda_update_scale),
        vmec2000_control=bool(vmec2000_control),
        lconm1=bool(lconm1),
        include_control_ptau=bool(include_control_ptau),
    )

    frcc = frzl_in.frcc
    fzsc = frzl_in.fzsc
    frss = frzl_in.frss if has_frss else frcc
    fzcs = frzl_in.fzcs if has_fzcs else fzsc
    frsc = getattr(frzl_in, "frsc", None) if has_frsc else frcc
    frcs = getattr(frzl_in, "frcs", None) if has_frcs else frcc
    fzcc = getattr(frzl_in, "fzcc", None) if has_fzcc else fzsc
    fzss = getattr(frzl_in, "fzss", None) if has_fzss else fzsc

    ar = mats["ar"]
    br = mats["br"]
    dr = mats["dr"]
    cr = mats.get("cr", ar)
    ir = mats.get("ir", dr)
    az = mats["az"]
    bz = mats["bz"]
    dz = mats["dz"]
    cz = mats.get("cz", az)
    iz = mats.get("iz", dz)
    dlr_t = mats.get("dlr_t", ar)
    dr_t = mats.get("dr_t", ar)
    dur_t = mats.get("dur_t", ar)
    dlz_t = mats.get("dlz_t", az)
    dz_t = mats.get("dz_t", az)
    duz_t = mats.get("duz_t", az)

    args = (
        frcc,
        fzsc,
        frss,
        fzcs,
        frsc,
        frcs,
        fzcc,
        fzss,
        ar,
        br,
        dr,
        cr,
        ir,
        az,
        bz,
        dz,
        cz,
        iz,
        dlr_t,
        dr_t,
        dur_t,
        dlz_t,
        dz_t,
        duz_t,
        frzl_in.flsc,
        frzl_in.flcs,
        getattr(frzl_in, "flcc", None),
        getattr(frzl_in, "flss", None),
        lam_prec,
        w_mode_mn,
        lambda_update_scale_j,
        f_norm1,
        delta_s,
        s,
    )
    if bool(include_control_ptau):
        args = (
            *args,
            *control_ptau_arrays,
            control_ptau_pshalf,
            control_ptau_ohs,
        )
    return apply_payload(*args)


def _mode_weight_force_blocks_np(
    blocks: _ForceBlocks,
    *,
    w_mode_mn,
    zeros_coeff,
) -> _ForceBlocks:
    """Scale preconditioned host force blocks by mode weights.

    Missing optional blocks intentionally reuse ``zeros_coeff`` so the hot
    host path avoids repeated zero-array allocations.
    """

    weight = np.asarray(w_mode_mn)[None, :, :]
    zero = zeros_coeff

    def _optional_scale(a):
        return np.asarray(a) * weight if a is not None else zero

    return _ForceBlocks(
        frcc=np.asarray(blocks.frcc) * weight,
        frss=_optional_scale(blocks.frss),
        fzsc=np.asarray(blocks.fzsc) * weight,
        fzcs=_optional_scale(blocks.fzcs),
        flsc=np.asarray(blocks.flsc) * weight,
        flcs=_optional_scale(blocks.flcs),
        frsc=_optional_scale(blocks.frsc),
        frcs=_optional_scale(blocks.frcs),
        fzcc=_optional_scale(blocks.fzcc),
        fzss=_optional_scale(blocks.fzss),
        flcc=_optional_scale(blocks.flcc),
        flss=_optional_scale(blocks.flss),
    )


def _lambda_preconditioned_full_norm(frzl_pre, *, use_jax: bool):
    """Return VMEC2000 full-mesh lambda preconditioned residual norm."""
    xp = jnp if bool(use_jax) else np
    flsc = xp.asarray(frzl_pre.flsc)
    gcl2_full = xp.sum(flsc[1:] * flsc[1:])
    if frzl_pre.flcs is not None:
        flcs = xp.asarray(frzl_pre.flcs)
        gcl2_full = gcl2_full + xp.sum(flcs[1:] * flcs[1:])
    if getattr(frzl_pre, "flcc", None) is not None:
        flcc = xp.asarray(frzl_pre.flcc)
        gcl2_full = gcl2_full + xp.sum(flcc[1:] * flcc[1:])
    if getattr(frzl_pre, "flss", None) is not None:
        flss = xp.asarray(frzl_pre.flss)
        gcl2_full = gcl2_full + xp.sum(flss[1:] * flss[1:])
    return gcl2_full


def _safe_dt_from_force_blocks(
    *,
    dt_nominal: float,
    max_coeff_delta_rms: float,
    blocks: _ForceBlocks,
) -> float:
    """Limit dt from force RMS when the optional stability guard is enabled."""
    frcc = jnp.asarray(blocks.frcc)
    frss = jnp.asarray(blocks.frss) if blocks.frss is not None else jnp.zeros_like(frcc)
    fzsc = jnp.asarray(blocks.fzsc)
    fzcs = jnp.asarray(blocks.fzcs) if blocks.fzcs is not None else jnp.zeros_like(fzsc)
    flsc = jnp.asarray(blocks.flsc)
    flcs = jnp.asarray(blocks.flcs) if blocks.flcs is not None else jnp.zeros_like(flsc)
    frsc = jnp.asarray(blocks.frsc) if blocks.frsc is not None else jnp.zeros_like(frcc)
    frcs = jnp.asarray(blocks.frcs) if blocks.frcs is not None else jnp.zeros_like(frcc)
    fzcc = jnp.asarray(blocks.fzcc) if blocks.fzcc is not None else jnp.zeros_like(fzsc)
    fzss = jnp.asarray(blocks.fzss) if blocks.fzss is not None else jnp.zeros_like(fzsc)
    flcc = jnp.asarray(blocks.flcc) if blocks.flcc is not None else jnp.zeros_like(flsc)
    flss = jnp.asarray(blocks.flss) if blocks.flss is not None else jnp.zeros_like(flsc)
    rms = jnp.sqrt(
        jnp.mean(
            frcc * frcc
            + frss * frss
            + frsc * frsc
            + frcs * frcs
            + fzsc * fzsc
            + fzcs * fzcs
            + fzcc * fzcc
            + fzss * fzss
            + flsc * flsc
            + flcs * flcs
            + flcc * flcc
            + flss * flss
        )
    )
    rms_f = float(np.asarray(rms))
    if not np.isfinite(rms_f) or rms_f <= 0.0:
        return max(float(dt_nominal), 1e-12)
    # With this integrator, first-step coefficient update is O(dt^2 * force).
    dt_lim = np.sqrt(float(max_coeff_delta_rms) / max(rms_f, 1e-30))
    dt_eff = min(float(dt_nominal), float(dt_lim))
    return max(dt_eff, 1e-12)


if has_jax():

    @jax.jit
    def _ptau_compute_jit(
        pru_even,
        pru_odd,
        pzu_even,
        pzu_odd,
        pr1_even,
        pr1_odd,
        pz1_even,
        pz1_odd,
        pshalf,
        ohs,
    ):
        """Compute ptau min/max without redefining a hot JIT helper per solve."""
        pshalf = pshalf.astype(pru_even.dtype)
        ohs = ohs.astype(pru_even.dtype)
        dphids = jnp.asarray(0.25, dtype=pru_even.dtype)
        psh = pshalf[1:][:, None, None]
        psh_safe = jnp.where(psh != 0.0, psh, jnp.ones_like(psh))
        ru12 = 0.5 * (pru_even[1:] + pru_even[:-1] + psh * (pru_odd[1:] + pru_odd[:-1]))
        pzs = ohs * ((pz1_even[1:] - pz1_even[:-1]) + psh * (pz1_odd[1:] - pz1_odd[:-1]))
        ptau = ru12 * pzs + dphids * (
            pru_odd[1:] * pz1_odd[1:]
            + pru_odd[:-1] * pz1_odd[:-1]
            + (pru_even[1:] * pz1_odd[1:] + pru_even[:-1] * pz1_odd[:-1]) / psh_safe
        )
        pzu12 = 0.5 * (pzu_even[1:] + pzu_even[:-1] + psh * (pzu_odd[1:] + pzu_odd[:-1]))
        prs = ohs * ((pr1_even[1:] - pr1_even[:-1]) + psh * (pr1_odd[1:] - pr1_odd[:-1]))
        ptau = (
            ptau
            - prs * pzu12
            - dphids
            * (
                pzu_odd[1:] * pr1_odd[1:]
                + pzu_odd[:-1] * pr1_odd[:-1]
                + (pzu_even[1:] * pr1_odd[1:] + pzu_even[:-1] * pr1_odd[:-1]) / psh_safe
            )
        )
        return jnp.min(ptau), jnp.max(ptau)
else:
    _ptau_compute_jit = None


_hash_array_bytes = _solve_runtime._hash_array_bytes
_tree_has_tracer = _solve_runtime._tree_has_tracer
_scan_backend_name = _solve_runtime._scan_backend_name
_parse_iter_list = _solve_runtime._parse_iter_list
_dump_env_enabled = _solve_runtime._dump_env_enabled
_dump_iter_selected = _solve_runtime._dump_iter_selected
_runtime_env_enabled = _solve_runtime._runtime_env_enabled
_edge_signature_key = _solve_runtime._edge_signature_key
_edge_value_key = _solve_runtime._edge_value_key
_scan_fallback_policy = _solve_runtime._scan_fallback_policy
_residual_convergence_flags = _solve_runtime._residual_convergence_flags
_scalar_history_array = _solve_runtime._scalar_history_array


def _scan_chunk_settings(
    *,
    max_iter_scan: int,
    nstep_screen: int,
    need_print: bool,
    lthreed: bool,
) -> tuple[int, bool]:
    return _resolve_scan_chunk_settings(
        max_iter_scan=max_iter_scan,
        nstep_screen=nstep_screen,
        need_print=need_print,
        lthreed=lthreed,
        backend_name=_scan_backend_name(),
        chunk_size_env=os.getenv("VMEC_JAX_SCAN_CHUNK_SIZE", ""),
    )


_default_scan_core = _solve_runtime._default_scan_core


def _radial_tridi_smooth_dirichlet(
    rhs,
    *,
    alpha: float,
    skip_nonpositive: bool = False,
    allow_3d: bool = True,
):
    """Solve the Dirichlet tri-diagonal smoothing system along the radial axis."""
    if skip_nonpositive and alpha <= 0.0:
        return rhs
    rhs = jnp.asarray(rhs)
    if rhs.ndim == 2:
        rhs2 = rhs
        orig_shape = None
    elif rhs.ndim < 2:
        raise ValueError(f"expected (ns,...) with ndim>=2, got {rhs.shape}")
    elif allow_3d and rhs.ndim == 3:
        ns = int(rhs.shape[0])
        rhs2 = rhs.reshape(ns, -1)
        orig_shape = rhs.shape
    elif allow_3d:
        raise ValueError(f"expected (ns,K) or (ns,M,N), got {rhs.shape}")
    else:
        raise ValueError(f"expected (ns,...) with ndim>=2, got {rhs.shape}")
    ns = int(rhs2.shape[0])
    if ns < 3:
        return rhs
    alpha_arr = jnp.asarray(alpha, dtype=rhs2.dtype)
    a = -alpha_arr
    b = 1.0 + 2.0 * alpha_arr
    c = -alpha_arr

    x0 = rhs2[0]
    xN = rhs2[-1]
    d = rhs2[1:-1]
    d = d.at[0].add(alpha_arr * x0)
    d = d.at[-1].add(alpha_arr * xN)

    n = int(d.shape[0])
    if n == 1:
        x_int = d / b
    else:
        cp0 = c / b
        dp0 = d[0] / b

        def fwd(carry, di):
            cp_prev, dp_prev = carry
            denom = b - a * cp_prev
            cp = c / denom
            dp = (di - a * dp_prev) / denom
            return (cp, dp), (cp, dp)

        (_cp_last, dp_last), (cp_rest, dp_rest) = jax.lax.scan(fwd, (cp0, dp0), d[1:])
        cp = jnp.concatenate([jnp.asarray([cp0]), cp_rest], axis=0)
        dp = jnp.concatenate([dp0[None, :], dp_rest], axis=0)

        def bwd(x_next, items):
            cpi, dpi = items
            xi = dpi - cpi * x_next
            return xi, xi

        _x0, x_rev = jax.lax.scan(bwd, dp_last, (cp[:-1], dp[:-1]), reverse=True)
        x_int = jnp.concatenate([x_rev, dp_last[None, :]], axis=0)

    out = jnp.concatenate([x0[None, :], x_int, xN[None, :]], axis=0)
    if orig_shape is not None:
        out = out.reshape(orig_shape)
    return out


def _metric_surface_precond_scales_jax(*, guu, r12, bsubu, bsubv, w_ang):
    """Approximate radial/lambda preconditioner scales with tracer-safe ops."""
    w3 = jnp.asarray(w_ang, dtype=jnp.asarray(guu).dtype)[None, :, :]
    rz_denom = jnp.sum((guu * (r12 * r12)) * w3, axis=(1, 2))
    rz_scale = jnp.where(rz_denom > 0.0, 1.0 / jnp.sqrt(jnp.maximum(rz_denom, 1e-300)), 1.0)
    l_denom = jnp.sum(((bsubu * bsubu) + (bsubv * bsubv)) * w3, axis=(1, 2))
    l_scale = jnp.where(l_denom > 0.0, 1.0 / jnp.sqrt(jnp.maximum(l_denom, 1e-300)), 1.0)
    return jnp.clip(rz_scale, 1e-4, 1e2), jnp.clip(l_scale, 1e-4, 1e2)


def _metric_surface_precond_scales_np(*, guu, r12, bsubu, bsubv, w_ang) -> tuple[np.ndarray, np.ndarray]:
    """Host NumPy variant of the first-step metric preconditioner scales."""
    guu_arr = np.asarray(guu)
    r12_arr = np.asarray(r12)
    bsubu_arr = np.asarray(bsubu)
    bsubv_arr = np.asarray(bsubv)
    w3 = np.asarray(w_ang, dtype=guu_arr.dtype)[None, :, :]
    rz_denom = np.sum((guu_arr * (r12_arr * r12_arr)) * w3, axis=(1, 2))
    rz_scale = np.where(rz_denom > 0.0, 1.0 / np.sqrt(np.maximum(rz_denom, 1e-300)), 1.0)
    l_denom = np.sum(((bsubu_arr * bsubu_arr) + (bsubv_arr * bsubv_arr)) * w3, axis=(1, 2))
    l_scale = np.where(l_denom > 0.0, 1.0 / np.sqrt(np.maximum(l_denom, 1e-300)), 1.0)
    return np.clip(rz_scale, 1e-4, 1e2), np.clip(l_scale, 1e-4, 1e2)


def _pshalf_from_s_np(s_arr) -> np.ndarray:
    s_arr = np.asarray(s_arr, dtype=float)
    if s_arr.size < 2:
        return np.sqrt(np.maximum(s_arr, 0.0))
    sh = 0.5 * (s_arr[1:] + s_arr[:-1])
    p = np.concatenate([sh[:1], sh], axis=0)
    return np.sqrt(np.maximum(p, 0.0))


def _pshalf_from_s_jax(s_arr, dtype):
    s_arr = jnp.asarray(s_arr, dtype=dtype)
    if int(s_arr.size) < 2:
        return jnp.sqrt(jnp.maximum(s_arr, jnp.asarray(0.0, dtype=dtype)))
    sh = 0.5 * (s_arr[1:] + s_arr[:-1])
    p = jnp.concatenate([sh[:1], sh], axis=0)
    return jnp.sqrt(jnp.maximum(p, jnp.asarray(0.0, dtype=dtype)))


def _sm_sp_from_s_np(s_arr) -> tuple[np.ndarray, np.ndarray]:
    s_arr = np.asarray(s_arr, dtype=float)
    ns = int(s_arr.shape[0])
    if ns < 2:
        z = np.zeros((ns + 1,), dtype=float)
        return z, z
    hs = s_arr[1] - s_arr[0]
    i = np.arange(ns + 1, dtype=float)
    psqrts = np.where(i >= 1, np.sqrt(np.maximum(hs * (i - 1.0), 0.0)), 0.0)
    psqrts[-1] = 1.0
    pshalf = np.where(i >= 1, np.sqrt(np.maximum(hs * np.abs(i - 1.5), 0.0)), 0.0)
    sm = np.zeros((ns + 1,), dtype=float)
    sp = np.zeros((ns + 1,), dtype=float)
    idx = np.arange(2, ns + 1)
    sm[idx] = np.where(psqrts[idx] != 0, pshalf[idx] / psqrts[idx], 0.0)
    sm[1] = 0.0
    idx2 = np.arange(2, ns)
    sp[idx2] = np.where(psqrts[idx2] != 0, pshalf[idx2 + 1] / psqrts[idx2], 0.0)
    sp[ns] = np.where(psqrts[ns] != 0, 1.0 / psqrts[ns], 0.0)
    sp[0] = 0.0
    sp[1] = sm[2] if ns >= 2 else 0.0
    return sm, sp


def _merge_axis_reset_state(*, st: VMECState, st_axis: VMECState, static, full_reset: bool) -> VMECState:
    """Return an axis-reset state, preserving non-axis coefficients unless full reset."""
    if full_reset:
        return st_axis
    if getattr(static, "m_is_m0", None) is None:
        mask_m0 = jnp.asarray(np.asarray(static.modes.m, dtype=int) == 0, dtype=jnp.asarray(st.Rcos).dtype)
    else:
        mask_m0 = jnp.asarray(static.m_is_m0, dtype=jnp.asarray(st.Rcos).dtype)
    Rcos = jnp.where(mask_m0[None, :] != 0, jnp.asarray(st_axis.Rcos), jnp.asarray(st.Rcos))
    Rsin = jnp.where(mask_m0[None, :] != 0, jnp.asarray(st_axis.Rsin), jnp.asarray(st.Rsin))
    Zcos = jnp.where(mask_m0[None, :] != 0, jnp.asarray(st_axis.Zcos), jnp.asarray(st.Zcos))
    Zsin = jnp.where(mask_m0[None, :] != 0, jnp.asarray(st_axis.Zsin), jnp.asarray(st.Zsin))
    return VMECState(
        layout=st.layout,
        Rcos=Rcos,
        Rsin=Rsin,
        Zcos=Zcos,
        Zsin=Zsin,
        Lcos=st.Lcos,
        Lsin=st.Lsin,
    )


@dataclass(frozen=True)
class SolveLambdaResult:
    state: VMECState
    n_iter: int
    wb_history: np.ndarray
    grad_rms_history: np.ndarray
    step_history: np.ndarray
    diagnostics: Dict[str, Any]


@dataclass(frozen=True)
class SolveFixedBoundaryResult:
    state: VMECState
    n_iter: int
    w_history: np.ndarray
    wb_history: np.ndarray
    wp_history: np.ndarray
    grad_rms_history: np.ndarray
    step_history: np.ndarray
    diagnostics: Dict[str, Any]


@dataclass(frozen=True)
class SolveVmecResidualResult:
    state: VMECState
    n_iter: int
    w_history: np.ndarray
    fsqr2_history: np.ndarray
    fsqz2_history: np.ndarray
    fsql2_history: np.ndarray
    grad_rms_history: np.ndarray
    step_history: np.ndarray
    diagnostics: Dict[str, Any]


class _ScanResumeInitialFields(NamedTuple):
    time_step: Any
    flip_sign: Any
    inv_tau: Any
    fsq_prev: Any
    fsq0_prev: Any
    res0: Any
    res1: Any
    iter1: Any
    ijacob: Any
    bad_resets: Any
    bad_growth: Any
    fsqz_prev: Any
    force_bcovar_update: Any
    vRcc: Any
    vRss: Any
    vZsc: Any
    vZcs: Any
    vLsc: Any
    vLcs: Any
    vRsc: Any
    vRcs: Any
    vZcc: Any
    vZss: Any
    vLcc: Any
    vLss: Any
    r00_prev: Any
    z00_prev: Any
    w_mhd_prev: Any
    state_checkpoint: Any


class _InitialAxisResetDecision(NamedTuple):
    bad_jacobian: bool
    force_reset: bool
    reset: bool


def _initial_axis_reset_decision(
    *,
    bad_jacobian_ptau: bool | None,
    bad_jacobian_state: bool,
    badjac_use_state: bool,
    fsq_phys: float | None,
    axis_reset_fsq_min: float,
    force_axis_reset: bool,
    axis_reset_always_3d: bool,
    lthreed: bool,
    vmec2000_control: bool = True,
    lmove_axis: bool = True,
    axis_reset_enabled: bool = True,
) -> _InitialAxisResetDecision:
    """Pure control-flow gate for VMEC-style initial magnetic-axis resets."""

    if bad_jacobian_ptau is None:
        bad_jacobian = bool(bad_jacobian_state)
    elif bool(badjac_use_state):
        bad_jacobian = bool(bad_jacobian_ptau) and bool(bad_jacobian_state)
    else:
        bad_jacobian = bool(bad_jacobian_ptau)

    fsq_min = max(0.0, float(axis_reset_fsq_min))
    if bad_jacobian and fsq_min > 0.0:
        if fsq_phys is None:
            bad_jacobian = False
        else:
            fsq_val = float(fsq_phys)
            if (not np.isfinite(fsq_val)) or (fsq_val < fsq_min):
                bad_jacobian = False

    force_reset = bool(force_axis_reset) or (
        bool(vmec2000_control) and bool(lmove_axis) and bool(lthreed) and bool(axis_reset_always_3d)
    )
    return _InitialAxisResetDecision(
        bad_jacobian=bool(bad_jacobian),
        force_reset=bool(force_reset),
        reset=bool(axis_reset_enabled) and (bool(bad_jacobian) or bool(force_reset)),
    )


def _write_axis_reset_dump(
    *,
    axis_dump_dir: str | os.PathLike[str] | None,
    ns: int,
    ntor: int,
    used_state_guess: bool,
    raxis_cc,
    raxis_cs,
    zaxis_cc,
    zaxis_cs,
) -> bool:
    """Write optional magnetic-axis reset coefficients for diagnostics."""

    if axis_dump_dir is None or str(axis_dump_dir).strip() == "":
        return False
    try:
        p = Path(axis_dump_dir).expanduser().resolve()
        ntor_i = int(ntor)
        rcc = np.asarray(raxis_cc)
        rcs = np.asarray(raxis_cs)
        zcc = np.asarray(zaxis_cc)
        zcs = np.asarray(zaxis_cs)
        if min(rcc.size, rcs.size, zcc.size, zcs.size) < ntor_i + 1:
            return False
        p.mkdir(parents=True, exist_ok=True)
        out = p / f"axis_reset_ns{int(ns)}.dat"
        with out.open("w", encoding="utf-8") as f:
            f.write(f"# used_state_guess={int(bool(used_state_guess))}\n")
            f.write("n raxis_cc raxis_cs zaxis_cc zaxis_cs\n")
            for n in range(ntor_i + 1):
                f.write(
                    f"{n:4d} "
                    f"{float(rcc[n]): .16e} "
                    f"{float(rcs[n]): .16e} "
                    f"{float(zcc[n]): .16e} "
                    f"{float(zcs[n]): .16e}\n"
                )
        return True
    except Exception:
        return False


def _initialize_scan_resume_state(
    resume_state: dict | None,
    *,
    dtype: Any,
    velocity_shape: tuple[int, ...],
    k_ndamp: int,
    time_step_default: Any,
    flip_sign_default: Any,
    state_checkpoint_default: Any,
) -> _ScanResumeInitialFields:
    time_step0 = jnp.asarray(time_step_default, dtype=dtype)
    flip_sign0 = jnp.asarray(flip_sign_default, dtype=dtype)
    inv_tau0 = jnp.full((k_ndamp,), jnp.asarray(0.15, dtype=dtype) / time_step0)
    fsq_prev0 = jnp.asarray(1.0, dtype=dtype)
    fsq0_prev0 = jnp.asarray(1.0, dtype=dtype)
    res0_0 = jnp.asarray(-1.0, dtype=dtype)
    res1_0 = jnp.asarray(-1.0, dtype=dtype)
    iter1_0 = jnp.asarray(1, dtype=jnp.int32)
    ijacob0 = jnp.asarray(0, dtype=jnp.int32)
    bad_resets0 = jnp.asarray(0, dtype=jnp.int32)
    bad_growth0 = jnp.asarray(0, dtype=jnp.int32)
    fsqz_prev0 = jnp.asarray(1.0, dtype=dtype)
    force_bcovar0 = jnp.asarray(False)

    vRcc0 = jnp.zeros(velocity_shape, dtype=dtype)
    vRss0 = jnp.zeros_like(vRcc0)
    vZsc0 = jnp.zeros_like(vRcc0)
    vZcs0 = jnp.zeros_like(vRcc0)
    vLsc0 = jnp.zeros_like(vRcc0)
    vLcs0 = jnp.zeros_like(vRcc0)
    vRsc0 = jnp.zeros_like(vRcc0)
    vRcs0 = jnp.zeros_like(vRcc0)
    vZcc0 = jnp.zeros_like(vRcc0)
    vZss0 = jnp.zeros_like(vRcc0)
    vLcc0 = jnp.zeros_like(vRcc0)
    vLss0 = jnp.zeros_like(vRcc0)
    r00_prev0 = jnp.asarray(0.0, dtype=dtype)
    z00_prev0 = jnp.asarray(0.0, dtype=dtype)
    w_mhd_prev0 = jnp.asarray(0.0, dtype=dtype)
    state_checkpoint0 = state_checkpoint_default

    if resume_state is not None:
        try:
            time_step0 = jnp.asarray(float(resume_state.get("time_step", time_step0)), dtype=dtype)
        except Exception:
            time_step0 = jnp.asarray(time_step0, dtype=dtype)
        try:
            flip_sign0 = jnp.asarray(float(resume_state.get("flip_sign", flip_sign0)), dtype=dtype)
        except Exception:
            pass
        inv_tau_val = resume_state.get("inv_tau", None)
        if inv_tau_val is not None:
            inv_tau0 = jnp.asarray(inv_tau_val, dtype=dtype)
        else:
            inv_tau0 = jnp.full((k_ndamp,), jnp.asarray(0.15, dtype=dtype) / time_step0)
        try:
            fsq_prev0 = jnp.asarray(float(resume_state.get("fsq_prev", fsq_prev0)), dtype=dtype)
        except Exception:
            pass
        try:
            fsq0_prev0 = jnp.asarray(float(resume_state.get("fsq0_prev", fsq0_prev0)), dtype=dtype)
        except Exception:
            pass
        try:
            res0_0 = jnp.asarray(float(resume_state.get("res0", res0_0)), dtype=dtype)
            res1_0 = jnp.asarray(float(resume_state.get("res1", res1_0)), dtype=dtype)
        except Exception:
            pass
        try:
            iter1_0 = jnp.asarray(int(resume_state.get("iter1", int(iter1_0))), dtype=jnp.int32)
        except Exception:
            pass
        try:
            ijacob0 = jnp.asarray(int(resume_state.get("ijacob", int(ijacob0))), dtype=jnp.int32)
        except Exception:
            pass
        try:
            bad_resets0 = jnp.asarray(int(resume_state.get("bad_resets", int(bad_resets0))), dtype=jnp.int32)
        except Exception:
            pass
        try:
            bad_growth0 = jnp.asarray(int(resume_state.get("bad_growth_streak", int(bad_growth0))), dtype=jnp.int32)
        except Exception:
            pass
        try:
            fsqz_prev0 = jnp.asarray(float(resume_state.get("fsqz_prev", fsqz_prev0)), dtype=dtype)
        except Exception:
            pass
        if "vRcc" in resume_state:
            vRcc0 = jnp.asarray(resume_state["vRcc"], dtype=dtype)
            vRss0 = jnp.asarray(resume_state.get("vRss", vRss0), dtype=dtype)
            vZsc0 = jnp.asarray(resume_state.get("vZsc", vZsc0), dtype=dtype)
            vZcs0 = jnp.asarray(resume_state.get("vZcs", vZcs0), dtype=dtype)
            vLsc0 = jnp.asarray(resume_state.get("vLsc", vLsc0), dtype=dtype)
            vLcs0 = jnp.asarray(resume_state.get("vLcs", vLcs0), dtype=dtype)
            vRsc0 = jnp.asarray(resume_state.get("vRsc", vRsc0), dtype=dtype)
            vRcs0 = jnp.asarray(resume_state.get("vRcs", vRcs0), dtype=dtype)
            vZcc0 = jnp.asarray(resume_state.get("vZcc", vZcc0), dtype=dtype)
            vZss0 = jnp.asarray(resume_state.get("vZss", vZss0), dtype=dtype)
            vLcc0 = jnp.asarray(resume_state.get("vLcc", vLcc0), dtype=dtype)
            vLss0 = jnp.asarray(resume_state.get("vLss", vLss0), dtype=dtype)
        try:
            force_bcovar0 = jnp.asarray(
                bool(resume_state.get("force_bcovar_update", bool(force_bcovar0))), dtype=bool
            )
        except Exception:
            pass
        if "r00_prev" in resume_state:
            r00_prev0 = jnp.asarray(resume_state.get("r00_prev", r00_prev0), dtype=dtype)
        if "z00_prev" in resume_state:
            z00_prev0 = jnp.asarray(resume_state.get("z00_prev", z00_prev0), dtype=dtype)
        if "w_mhd_prev" in resume_state:
            w_mhd_prev0 = jnp.asarray(resume_state.get("w_mhd_prev", w_mhd_prev0), dtype=dtype)
        state_checkpoint0 = resume_state.get("state_checkpoint", state_checkpoint0)

    return _ScanResumeInitialFields(
        time_step=time_step0,
        flip_sign=flip_sign0,
        inv_tau=inv_tau0,
        fsq_prev=fsq_prev0,
        fsq0_prev=fsq0_prev0,
        res0=res0_0,
        res1=res1_0,
        iter1=iter1_0,
        ijacob=ijacob0,
        bad_resets=bad_resets0,
        bad_growth=bad_growth0,
        fsqz_prev=fsqz_prev0,
        force_bcovar_update=force_bcovar0,
        vRcc=vRcc0,
        vRss=vRss0,
        vZsc=vZsc0,
        vZcs=vZcs0,
        vLsc=vLsc0,
        vLcs=vLcs0,
        vRsc=vRsc0,
        vRcs=vRcs0,
        vZcc=vZcc0,
        vZss=vZss0,
        vLcc=vLcc0,
        vLss=vLss0,
        r00_prev=r00_prev0,
        z00_prev=z00_prev0,
        w_mhd_prev=w_mhd_prev0,
        state_checkpoint=state_checkpoint0,
    )


class _ScanCarry(NamedTuple):
    state: VMECState
    time_step: Any
    inv_tau: Any
    fsq_prev: Any
    fsq0_prev: Any
    accepted_count: Any
    probe_count: Any
    probe_bad_jac: Any
    probe_accept: Any
    probe_fsq_min: Any
    probe_fsq_max: Any
    probe_fsq_start: Any
    fallback_active: Any
    abort_scan: Any
    skip_timecontrol: Any
    vRcc: Any
    vRss: Any
    vZsc: Any
    vZcs: Any
    vLsc: Any
    vLcs: Any
    vRsc: Any
    vRcs: Any
    vZcc: Any
    vZss: Any
    vLcc: Any
    vLss: Any
    flip_sign: Any
    iter_offset: Any
    iter1: Any
    res0: Any
    res1: Any
    state_checkpoint: VMECState
    cache_valid: Any
    cache_precond_diag: Any
    cache_tcon: Any
    cache_norms: Any
    cache_rz_scale: Any
    cache_l_scale: Any
    cache_rz_norm: Any
    cache_f_norm1: Any
    cache_prec_rz_mats: Any
    cache_prec_lam_prec: Any
    force_bcovar_update: Any
    ijacob: Any
    bad_resets: Any
    bad_growth: Any
    fsqz_prev: Any
    r00_prev: Any
    z00_prev: Any
    w_mhd_prev: Any
    converged: Any
    fsqr_prev_phys: Any
    fsqz_prev_phys: Any
    fsql_prev_phys: Any
    fsqr1_prev: Any
    fsqz1_prev: Any
    fsql1_prev: Any
    fsqr_checkpoint: Any
    fsqz_checkpoint: Any
    fsql_checkpoint: Any
    fsqr1_checkpoint: Any
    fsqz1_checkpoint: Any
    fsql1_checkpoint: Any
    edge_Rcos: Any
    edge_Rsin: Any
    edge_Zcos: Any
    edge_Zsin: Any


def _free_boundary_iter_controls(iter2: int, iter1: int, nvacskip: int) -> tuple[int, int]:
    """Simple free-boundary cadence helper (legacy/diagnostics path).

    Returns `(ivac, ivacskip)` from integer iteration counters using the
    reduced two-state model:
    - `ivacskip = mod(iter2 - iter1, nvacskip)`
    - `ivac = 1` for full vacuum update (`ivacskip==0`), else `2`.
    """

    nv = max(1, int(nvacskip))
    ivs = int((int(iter2) - int(iter1)) % nv)
    ivac = 1 if ivs == 0 else 2
    return ivac, ivs


def _free_boundary_iter_controls_vmec(
    *,
    iter2: int,
    iter1: int,
    ivac: int,
    nvacskip: int,
    nvskip0: int,
    fsq_rz_prev: float,
    activate_fsq: float | None = None,
) -> tuple[int, int, int]:
    """VMEC2000-style `ivac/ivacskip/nvacskip` update (funct3d, ictrl_prec2d=0).

    Mirrors funct3d.f free-boundary cadence (ictrl_prec2d=0):
    - `ivac` advances only when `(fsqr+fsqz) <= threshold`,
    - `ivac<=2` forces full vacuum updates (`ivacskip=0`),
    - `nvacskip` is adapted on full updates.
    """

    i2 = int(iter2)
    i1 = int(iter1)
    iv = int(ivac)
    nv = max(1, int(nvacskip))
    nv0 = max(1, int(nvskip0))
    fs = float(fsq_rz_prev)
    if not np.isfinite(fs) or fs < 0.0:
        fs = 1.0

    if activate_fsq is None:
        activate_threshold = float(os.getenv("VMEC_JAX_FREEB_ACTIVATE_FSQ", "1.0e-3") or 1.0e-3)
    else:
        activate_threshold = float(activate_fsq)
    # VMEC funct3d:
    #   IF (iter2 > 1 .AND. fsqr+fsqz <= 1e-3) ivac = ivac + 1
    # Keep this literal behavior for cadence parity.
    if i2 > 1 and fs <= activate_threshold:
        iv += 1

    if iv < 0:
        return iv, 0, nv

    ivs = int((i2 - i1) % nv)
    if iv <= 2:
        ivs = 0

    # Extend NVACSKIP as equilibrium converges, only on full update.
    if ivs == 0:
        nv_est = int(1.0 / max(1.0e-1, 1.0e11 * fs))
        nv = max(nv0, max(1, nv_est))

    return iv, ivs, nv


def _free_boundary_prev_rz_fsq_next(
    *,
    prev_fsq_before: float,
    fsq_rz_curr: float,
    turnon_restart: bool,
    preserve_turnon_restart: bool,
) -> float:
    """Optionally carry the pre-turn-on residual into the next cadence step."""
    if bool(turnon_restart) and bool(preserve_turnon_restart):
        return float(prev_fsq_before)
    return float(fsq_rz_curr)


def _free_boundary_should_damp_constraint_baseline(*, freeb_ivac: int, freeb_turnon_iter: bool, lthreed: bool) -> bool:
    """Mirror VMEC turn-on behavior for persistent `rcon0/zcon0` baselines.

    The first vacuum-coupled iteration should preserve the pre-turn-on
    constraint baseline on the 3D path. Axisymmetric turn-on keeps the
    original VMEC-aligned immediate damping behavior.
    """
    if not bool(lthreed):
        return int(freeb_ivac) >= 0
    return int(freeb_ivac) >= 0 and (not bool(freeb_turnon_iter))


def _free_boundary_turnon_resets_iter1_immediately(*, lthreed: bool, lasym: bool) -> bool:
    """Return whether turn-on should immediately reset `iter1` for cadence."""
    return (not bool(lthreed)) or (not bool(lasym))


def _zero_velocity_blocks_like(*blocks):
    """Return zeroed velocity blocks with each input block's shape and dtype."""

    out = []
    for block in blocks:
        if _tree_has_tracer(block):
            out.append(jnp.zeros_like(block))
            continue
        try:
            if jax is not None and isinstance(block, jax.Array):
                out.append(jnp.zeros_like(block))
                continue
        except Exception:
            pass
        out.append(np.zeros_like(np.asarray(block)))
    return tuple(out)


def _scale_velocity_blocks(scale: float, *blocks):
    """Scale velocity blocks uniformly while preserving JAX array semantics."""

    return tuple(float(scale) * block for block in blocks)


def _sample_free_boundary_external_field(*, state: VMECState, static) -> dict[str, Any]:
    """WP2 diagnostic scaffold for external-field boundary channels."""
    from .free_boundary import sample_external_vacuum_diagnostics

    plascur = float(getattr(static, "free_boundary_plascur", 0.0) or 0.0)
    return sample_external_vacuum_diagnostics(
        state=state,
        static=static,
        plascur=plascur,
    )


def _maybe_dump_tomnsps(*, frzl, static, iter_idx: int, label: str = "raw") -> None:
    env = os.getenv("VMEC_JAX_DUMP_TOMNSPS", "")
    if not _dump_env_enabled(env):
        return
    if not _dump_iter_selected(iter_idx=iter_idx, iter_env=os.getenv("VMEC_JAX_DUMP_ITER", "")):
        return
    outdir = Path(os.getenv("VMEC_JAX_DUMP_DIR", ".")).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    ns = int(static.cfg.ns)
    path = outdir / f"tomnsps_{label}_ns{ns}_iter{int(iter_idx)}.npz"

    def _arr(x):
        return np.asarray(x) if x is not None else np.zeros((0,), dtype=float)

    np.savez(
        path,
        frcc=_arr(frzl.frcc),
        frss=_arr(getattr(frzl, "frss", None)),
        fzsc=_arr(frzl.fzsc),
        fzcs=_arr(getattr(frzl, "fzcs", None)),
        flsc=_arr(frzl.flsc),
        flcs=_arr(getattr(frzl, "flcs", None)),
        frsc=_arr(getattr(frzl, "frsc", None)),
        frcs=_arr(getattr(frzl, "frcs", None)),
        fzcc=_arr(getattr(frzl, "fzcc", None)),
        fzss=_arr(getattr(frzl, "fzss", None)),
        flcc=_arr(getattr(frzl, "flcc", None)),
        flss=_arr(getattr(frzl, "flss", None)),
        ns=int(static.cfg.ns),
        mpol=int(static.cfg.mpol),
        ntor=int(static.cfg.ntor),
        lasym=bool(static.cfg.lasym),
    )


def _maybe_dump_force_kernels(*, k, static, iter_idx: int, label: str = "raw") -> None:
    env = os.getenv("VMEC_JAX_DUMP_FORCE_KERNELS", "")
    if not _dump_env_enabled(env):
        return
    if not _dump_iter_selected(iter_idx=iter_idx, iter_env=os.getenv("VMEC_JAX_DUMP_ITER", "")):
        return
    outdir = Path(os.getenv("VMEC_JAX_DUMP_DIR", ".")).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    ns = int(static.cfg.ns)
    path = outdir / f"force_kernels_{label}_ns{ns}_iter{int(iter_idx)}.npz"

    def _arr(x):
        return np.asarray(x) if x is not None else np.zeros((0,), dtype=float)

    np.savez(
        path,
        armn_e=_arr(getattr(k, "armn_e", None)),
        armn_o=_arr(getattr(k, "armn_o", None)),
        brmn_e=_arr(getattr(k, "brmn_e", None)),
        brmn_o=_arr(getattr(k, "brmn_o", None)),
        crmn_e=_arr(getattr(k, "crmn_e", None)),
        crmn_o=_arr(getattr(k, "crmn_o", None)),
        azmn_e=_arr(getattr(k, "azmn_e", None)),
        azmn_o=_arr(getattr(k, "azmn_o", None)),
        bzmn_e=_arr(getattr(k, "bzmn_e", None)),
        bzmn_o=_arr(getattr(k, "bzmn_o", None)),
        czmn_e=_arr(getattr(k, "czmn_e", None)),
        czmn_o=_arr(getattr(k, "czmn_o", None)),
        arcon_e=_arr(getattr(k, "arcon_e", None)),
        arcon_o=_arr(getattr(k, "arcon_o", None)),
        azcon_e=_arr(getattr(k, "azcon_e", None)),
        azcon_o=_arr(getattr(k, "azcon_o", None)),
        gcon=_arr(getattr(k, "gcon", None)),
        tcon=_arr(getattr(k, "tcon", None)),
        blmn_e=_arr(getattr(getattr(k, "bc", None), "blmn_even", None)),
        blmn_o=_arr(getattr(getattr(k, "bc", None), "blmn_odd", None)),
        clmn_e=_arr(getattr(getattr(k, "bc", None), "clmn_even", None)),
        clmn_o=_arr(getattr(getattr(k, "bc", None), "clmn_odd", None)),
        bsubu_e=_arr(getattr(getattr(k, "bc", None), "bsubu_e", None)),
        bsubv_e=_arr(getattr(getattr(k, "bc", None), "bsubv_e", None)),
        bsubu=_arr(getattr(getattr(k, "bc", None), "bsubu", None)),
        bsubv=_arr(getattr(getattr(k, "bc", None), "bsubv", None)),
        bsupu=_arr(getattr(getattr(k, "bc", None), "bsupu", None)),
        bsupv=_arr(getattr(getattr(k, "bc", None), "bsupv", None)),
        guu_metric=_arr(getattr(getattr(k, "bc", None), "guu", None)),
        guv_metric=_arr(getattr(getattr(k, "bc", None), "guv", None)),
        gvv_metric=_arr(getattr(getattr(k, "bc", None), "gvv", None)),
        sqrtg=_arr(getattr(getattr(getattr(k, "bc", None), "jac", None), "sqrtg", None)),
        r12=_arr(getattr(getattr(getattr(k, "bc", None), "jac", None), "r12", None)),
        tau=_arr(getattr(getattr(getattr(k, "bc", None), "jac", None), "tau", None)),
        ru12=_arr(getattr(getattr(getattr(k, "bc", None), "jac", None), "ru12", None)),
        zu12=_arr(getattr(getattr(getattr(k, "bc", None), "jac", None), "zu12", None)),
        rs=_arr(getattr(getattr(getattr(k, "bc", None), "jac", None), "rs", None)),
        zs=_arr(getattr(getattr(getattr(k, "bc", None), "jac", None), "zs", None)),
        bsubu_e_scaled=_arr(
            getattr(getattr(k, "bc", None), "bsubu_e_scaled", None)
            if getattr(getattr(k, "bc", None), "bsubu_e_scaled", None) is not None
            else getattr(getattr(k, "bc", None), "clmn_even", None)
        ),
        bsubv_e_scaled=_arr(
            getattr(getattr(k, "bc", None), "bsubv_e_scaled", None)
            if getattr(getattr(k, "bc", None), "bsubv_e_scaled", None) is not None
            else getattr(getattr(k, "bc", None), "blmn_even", None)
        ),
        bsubu_tmp=_arr(getattr(getattr(k, "bc", None), "bsubu_tmp", None)),
        bsubv_preblend=_arr(getattr(getattr(k, "bc", None), "bsubv_preblend", None)),
        bsubv_avg=_arr(getattr(getattr(k, "bc", None), "bsubv_avg", None)),
        lamscale=_arr(getattr(getattr(k, "bc", None), "lamscale", None)),
        lu0_full=_arr(getattr(getattr(k, "bc", None), "lu0_full", None)),
        lu0_force=_arr(getattr(getattr(k, "bc", None), "lu0_force", None)),
        lu1_full=_arr(getattr(getattr(k, "bc", None), "lu1_full", None)),
        lvv=_arr(getattr(getattr(k, "bc", None), "lvv", None)),
        lvv_sh=_arr(getattr(getattr(k, "bc", None), "lvv_sh", None)),
        phip_full=_arr(getattr(getattr(k, "bc", None), "phip_full", None)),
        phip_internal=_arr(getattr(getattr(k, "bc", None), "phip_internal", None)),
        pr1_even=_arr(getattr(k, "pr1_even", None)),
        pr1_odd=_arr(getattr(k, "pr1_odd", None)),
        pz1_even=_arr(getattr(k, "pz1_even", None)),
        pz1_odd=_arr(getattr(k, "pz1_odd", None)),
        pru_even=_arr(getattr(k, "pru_even", None)),
        pru_odd=_arr(getattr(k, "pru_odd", None)),
        pzu_even=_arr(getattr(k, "pzu_even", None)),
        pzu_odd=_arr(getattr(k, "pzu_odd", None)),
        prv_even=_arr(getattr(k, "prv_even", None)),
        prv_odd=_arr(getattr(k, "prv_odd", None)),
        pzv_even=_arr(getattr(k, "pzv_even", None)),
        pzv_odd=_arr(getattr(k, "pzv_odd", None)),
        ns=int(static.cfg.ns),
        ntheta=int(static.cfg.ntheta),
        nzeta=int(static.cfg.nzeta),
        lasym=bool(static.cfg.lasym),
    )


def _maybe_dump_scalars(*, norms, iter_idx: int, ns: int) -> None:
    env = os.getenv("VMEC_JAX_DUMP_SCALARS", "")
    if not _dump_env_enabled(env):
        return
    if not _dump_iter_selected(iter_idx=iter_idx, iter_env=os.getenv("VMEC_JAX_DUMP_ITER", "")):
        return
    outdir = Path(os.getenv("VMEC_JAX_DUMP_DIR", ".")).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"scalars_ns{int(ns)}_iter{int(iter_idx)}.dat"

    wb = float(np.asarray(getattr(norms, "wb", np.nan)))
    wp = float(np.asarray(getattr(norms, "wp", np.nan)))
    volume = float(np.asarray(getattr(norms, "volume", np.nan)))
    r2 = float(np.asarray(getattr(norms, "r2", np.nan)))
    fnorm = float(np.asarray(getattr(norms, "fnorm", np.nan)))
    fnormL = float(np.asarray(getattr(norms, "fnormL", np.nan)))
    fnorm1 = float("nan")
    with path.open("w") as f:
        f.write("# bcovar scalars dump\n")
        f.write("cols: iter wb wp vol r2 fnorm\n")
        f.write("      fn1 fnL\n")
        f.write(
            f"{int(iter_idx):6d}"
            f"{wb:24.16e}{wp:24.16e}{volume:24.16e}{r2:24.16e}"
            f"{fnorm:24.16e}{fnorm1:24.16e}{fnormL:24.16e}\n"
        )


def _maybe_dump_gcx2(*, gcr2, gcz2, gcl2, iter_idx: int, include_edge: bool, ns: int) -> None:
    env = os.getenv("VMEC_JAX_DUMP_GCX2", "")
    if not _dump_env_enabled(env):
        return
    if not _dump_iter_selected(iter_idx=iter_idx, iter_env=os.getenv("VMEC_JAX_DUMP_ITER", "")):
        return
    outdir = Path(os.getenv("VMEC_JAX_DUMP_DIR", ".")).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"gcx2_ns{int(ns)}_iter{int(iter_idx)}.dat"
    with path.open("w") as f:
        f.write("# gcx2 dump (post-scalxc, post-m1)\n")
        f.write("columns: iter include_edge gcr2 gcz2 gcl2\n")
        f.write(
            f"{int(iter_idx):6d} {int(bool(include_edge)):3d}"
            f"{float(np.asarray(gcr2)):24.16e}"
            f"{float(np.asarray(gcz2)):24.16e}"
            f"{float(np.asarray(gcl2)):24.16e}\n"
        )


def _gc_from_frzl(*, frzl, cfg):
    frcc = np.asarray(frzl.frcc)
    ns, mpol, nrange = frcc.shape
    lthreed = bool(getattr(cfg, "lthreed", True))
    lasym = bool(getattr(cfg, "lasym", False))
    if lasym:
        ntmax = 4 if lthreed else 2
    else:
        ntmax = 2 if lthreed else 1

    gcr = np.zeros((ns, mpol, nrange, ntmax), dtype=frcc.dtype)
    gcz = np.zeros_like(gcr)
    gcl = np.zeros_like(gcr)

    gcr[:, :, :, 0] = frcc
    gcz[:, :, :, 0] = np.asarray(frzl.fzsc)
    gcl[:, :, :, 0] = np.asarray(frzl.flsc)

    if lasym:
        if lthreed:
            if frzl.frss is not None:
                gcr[:, :, :, 1] = np.asarray(frzl.frss)
            if frzl.fzcs is not None:
                gcz[:, :, :, 1] = np.asarray(frzl.fzcs)
            if frzl.flcs is not None:
                gcl[:, :, :, 1] = np.asarray(frzl.flcs)
            if getattr(frzl, "frsc", None) is not None:
                gcr[:, :, :, 2] = np.asarray(frzl.frsc)
            if getattr(frzl, "fzcc", None) is not None:
                gcz[:, :, :, 2] = np.asarray(frzl.fzcc)
            if getattr(frzl, "flcc", None) is not None:
                gcl[:, :, :, 2] = np.asarray(frzl.flcc)
            if getattr(frzl, "frcs", None) is not None:
                gcr[:, :, :, 3] = np.asarray(frzl.frcs)
            if getattr(frzl, "fzss", None) is not None:
                gcz[:, :, :, 3] = np.asarray(frzl.fzss)
            if getattr(frzl, "flss", None) is not None:
                gcl[:, :, :, 3] = np.asarray(frzl.flss)
        else:
            if getattr(frzl, "frsc", None) is not None:
                gcr[:, :, :, 1] = np.asarray(frzl.frsc)
            if getattr(frzl, "fzcc", None) is not None:
                gcz[:, :, :, 1] = np.asarray(frzl.fzcc)
            if getattr(frzl, "flcc", None) is not None:
                gcl[:, :, :, 1] = np.asarray(frzl.flcc)
    else:
        if lthreed:
            if frzl.frss is not None:
                gcr[:, :, :, 1] = np.asarray(frzl.frss)
            if frzl.fzcs is not None:
                gcz[:, :, :, 1] = np.asarray(frzl.fzcs)
            if frzl.flcs is not None:
                gcl[:, :, :, 1] = np.asarray(frzl.flcs)

    return gcr, gcz, gcl


def _maybe_dump_gc(*, frzl, static, iter_idx: int, label: str) -> None:
    env = os.getenv("VMEC_JAX_DUMP_GC", "")
    if not env or env == "0":
        return
    iters = _parse_iter_list(os.getenv("VMEC_JAX_DUMP_GC_ITER", ""))
    if iters is not None and int(iter_idx) not in iters:
        return
    stage = os.getenv("VMEC_JAX_DUMP_GC_STAGE", "precond").lower()
    if stage not in {"raw", "precond", "both"}:
        stage = "precond"
    if stage != "both" and stage != label:
        return

    outdir = Path(os.getenv("VMEC_JAX_DUMP_GC_DIR", ".")).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    ns = int(static.cfg.ns)
    path = outdir / f"gc_{label}_ns{ns}_iter{int(iter_idx)}.npz"
    gcr, gcz, gcl = _gc_from_frzl(frzl=frzl, cfg=static.cfg)
    np.savez(
        path,
        gcr=gcr,
        gcz=gcz,
        gcl=gcl,
        ns=int(static.cfg.ns),
        mpol=int(static.cfg.mpol),
        ntor=int(static.cfg.ntor),
        lthreed=bool(static.cfg.lthreed),
        lasym=bool(static.cfg.lasym),
    )


def _maybe_dump_lam_prec(*, lam_prec, faclam, static, iter_idx: int) -> None:
    env = os.getenv("VMEC_JAX_DUMP_LAM", "")
    if not env or env == "0":
        return
    iters = _parse_iter_list(os.getenv("VMEC_JAX_DUMP_LAM_ITER", ""))
    if iters is None:
        iters = _parse_iter_list(os.getenv("VMEC_JAX_DUMP_ITER", ""))
    if iters is not None and int(iter_idx) not in iters:
        return
    outdir = os.getenv("VMEC_JAX_DUMP_LAM_DIR", "")
    if not outdir:
        outdir = os.getenv("VMEC_JAX_DUMP_DIR", ".")
    outdir = Path(outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    ns = int(static.cfg.ns)
    path = outdir / f"lam_prec_ns{ns}_iter{int(iter_idx)}.npz"
    lthreed = bool(static.cfg.lthreed)
    lasym = bool(static.cfg.lasym)
    if lasym:
        ntmax = 4 if lthreed else 2
    else:
        ntmax = 2 if lthreed else 1
    lam_arr = np.asarray(lam_prec)
    if lam_arr.ndim != 3:
        raise ValueError(f"lam_prec expected 3D (ns,mpol,ntor+1), got {lam_arr.shape}")
    # VMEC dumps use (ns, n, m, t) with t=1..ntmax.
    pfaclam = np.zeros((ns, lam_arr.shape[2], lam_arr.shape[1], ntmax), dtype=lam_arr.dtype)
    pfaclam[:, :, :, 0] = np.transpose(lam_arr, (0, 2, 1))
    if ntmax > 1:
        pfaclam[:, :, :, 1:ntmax] = pfaclam[:, :, :, :1]
        # VMEC updates (m,n)=(0,0) only for t=1, leaving t>1 at zero.
        pfaclam[:, 0, 0, 1:ntmax] = 0.0
    data = {
        "pfaclam": pfaclam,
        "ns": ns,
        "mpol": int(static.cfg.mpol),
        "ntor": int(static.cfg.ntor),
        "lthreed": lthreed,
        "lasym": lasym,
    }
    if faclam is not None:
        fac_arr = np.asarray(faclam)
        faclam_out = np.zeros_like(pfaclam)
        if fac_arr.shape == lam_arr.shape:
            faclam_out[:, :, :, 0] = np.transpose(fac_arr, (0, 2, 1))
            if ntmax > 1:
                faclam_out[:, :, :, 1:ntmax] = faclam_out[:, :, :, :1]
                faclam_out[:, 0, 0, 1:ntmax] = 0.0
        else:
            faclam_out = fac_arr
        data["faclam"] = faclam_out
    np.savez(path, **data)


def _maybe_dump_precond_mats(*, mats, static, iter_idx: int, jmax: int, used_cache: bool | None = None) -> None:
    env = os.getenv("VMEC_JAX_DUMP_PRECOND_MATS", "")
    if not env or env == "0":
        return
    iters = _parse_iter_list(os.getenv("VMEC_JAX_DUMP_PRECOND_MATS_ITER", ""))
    if iters is None:
        iters = _parse_iter_list(os.getenv("VMEC_JAX_DUMP_ITER", ""))
    if iters is not None and int(iter_idx) not in iters:
        return
    outdir = os.getenv("VMEC_JAX_DUMP_PRECOND_MATS_DIR", "")
    if not outdir:
        outdir = os.getenv("VMEC_JAX_DUMP_DIR", ".")
    outdir = Path(outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    ns = int(static.cfg.ns)
    path = outdir / f"precond_mats_ns{ns}_iter{int(iter_idx)}.npz"
    data = {
        "ns": ns,
        "mpol": int(static.cfg.mpol),
        "ntor": int(static.cfg.ntor),
        "lthreed": bool(static.cfg.lthreed),
        "lasym": bool(static.cfg.lasym),
        "jmax": int(jmax),
    }
    if used_cache is not None:
        data["used_cache"] = bool(used_cache)
    for key in ("ar", "br", "dr", "az", "bz", "dz"):
        if key in mats:
            data[key] = np.asarray(mats[key])
    np.savez(path, **data)


def _maybe_dump_lam_fsql1(*, fsql1_pre, fsql1_post, static, iter_idx: int) -> None:
    env = os.getenv("VMEC_JAX_DUMP_LAM", "")
    if not env or env == "0":
        return
    iters = _parse_iter_list(os.getenv("VMEC_JAX_DUMP_LAM_ITER", ""))
    if iters is None:
        iters = _parse_iter_list(os.getenv("VMEC_JAX_DUMP_ITER", ""))
    if iters is not None and int(iter_idx) not in iters:
        return
    outdir = os.getenv("VMEC_JAX_DUMP_LAM_DIR", "")
    if not outdir:
        outdir = os.getenv("VMEC_JAX_DUMP_DIR", ".")
    outdir = Path(outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    ns = int(static.cfg.ns)
    path = outdir / f"lam_fsql1_ns{ns}_iter{int(iter_idx)}.dat"
    with path.open("w", encoding="utf-8") as f:
        f.write("# lambda fsql1 dump (pre/post faclam)\n")
        f.write("columns: iter fsql1_pre fsql1_post\n")
        f.write(f"{int(iter_idx):6d} {float(np.asarray(fsql1_pre)):24.16e} {float(np.asarray(fsql1_post)):24.16e}\n")


def _maybe_dump_lamcal(*, lam_debug: dict[str, np.ndarray], static, iter_idx: int) -> None:
    env = os.getenv("VMEC_JAX_DUMP_LAMCAL", "")
    if not env or env == "0":
        return
    iters = _parse_iter_list(os.getenv("VMEC_JAX_DUMP_ITER", ""))
    if iters is not None and int(iter_idx) not in iters:
        return
    outdir = Path(os.getenv("VMEC_JAX_DUMP_DIR", ".")).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    ns = int(static.cfg.ns)
    path = outdir / f"lamcal_ns{ns}_iter{int(iter_idx)}.npz"
    np.savez(
        path,
        blam_pre=np.asarray(lam_debug.get("blam_pre")),
        clam_pre=np.asarray(lam_debug.get("clam_pre")),
        dlam_pre=np.asarray(lam_debug.get("dlam_pre")),
        blam_post=np.asarray(lam_debug.get("blam_post")),
        clam_post=np.asarray(lam_debug.get("clam_post")),
        dlam_post=np.asarray(lam_debug.get("dlam_post")),
    )


def _vmec_scale_m1_factors_from_mats(mats: dict[str, Any]) -> tuple[Any, Any]:
    """Return VMEC `scale_m1_par` R/Z factors from cached preconditioner data."""
    ard = mats.get("ard_parity")
    brd = mats.get("brd_parity")
    azd = mats.get("azd_parity")
    bzd = mats.get("bzd_parity")
    if ard is not None and brd is not None and azd is not None and bzd is not None:
        ard_arr = jnp.asarray(ard)
        brd_arr = jnp.asarray(brd)
        azd_arr = jnp.asarray(azd)
        bzd_arr = jnp.asarray(bzd)
        if (
            ard_arr.ndim == 2
            and brd_arr.shape == ard_arr.shape
            and azd_arr.shape == ard_arr.shape
            and bzd_arr.shape == ard_arr.shape
            and ard_arr.shape[1] > 1
        ):
            sr = ard_arr[:, 1] + brd_arr[:, 1]
            sz = azd_arr[:, 1] + bzd_arr[:, 1]
            denom = sr + sz
            fac_r = jnp.where(denom != 0.0, sr / denom, 1.0)
            fac_z = jnp.where(denom != 0.0, sz / denom, 1.0)
            return fac_r, fac_z

    dr = jnp.asarray(mats["dr"])
    dz = jnp.asarray(mats["dz"])
    sr = -dr[:, 1, 0]
    sz = -dz[:, 1, 0]
    denom = sr + sz
    fac_r = jnp.where(denom != 0.0, sr / denom, 1.0)
    fac_z = jnp.where(denom != 0.0, sz / denom, 1.0)
    return fac_r, fac_z


def _vmec_scale_m1_factors_from_mats_np(mats: dict) -> tuple[np.ndarray, np.ndarray]:
    """NumPy version of _vmec_scale_m1_factors_from_mats (no JAX dispatch)."""
    ard = mats.get("ard_parity")
    brd = mats.get("brd_parity")
    azd = mats.get("azd_parity")
    bzd = mats.get("bzd_parity")
    if ard is not None and brd is not None and azd is not None and bzd is not None:
        ard_arr = np.asarray(ard)
        brd_arr = np.asarray(brd)
        azd_arr = np.asarray(azd)
        bzd_arr = np.asarray(bzd)
        if (
            ard_arr.ndim == 2
            and brd_arr.shape == ard_arr.shape
            and azd_arr.shape == ard_arr.shape
            and bzd_arr.shape == ard_arr.shape
            and ard_arr.shape[1] > 1
        ):
            sr = ard_arr[:, 1] + brd_arr[:, 1]
            sz = azd_arr[:, 1] + bzd_arr[:, 1]
            denom = sr + sz
            fac_r = np.where(denom != 0.0, sr / np.where(denom != 0.0, denom, 1.0), 1.0)
            fac_z = np.where(denom != 0.0, sz / np.where(denom != 0.0, denom, 1.0), 1.0)
            return fac_r, fac_z
    dr = np.asarray(mats["dr"])
    dz = np.asarray(mats["dz"])
    sr = -dr[:, 1, 0]
    sz = -dz[:, 1, 0]
    denom = sr + sz
    fac_r = np.where(denom != 0.0, sr / np.where(denom != 0.0, denom, 1.0), 1.0)
    fac_z = np.where(denom != 0.0, sz / np.where(denom != 0.0, denom, 1.0), 1.0)
    return fac_r, fac_z


def _scale_m1_precond_rhs_from_mats(
    frzl_in,
    mats: dict[str, Any],
    *,
    lconm1: bool,
    mpol: int,
    host_update_assembly: bool,
):
    """Apply VMEC `scale_m1_par` factors before the radial preconditioner solve."""
    if (not bool(lconm1)) or (int(mpol) <= 1):
        return frzl_in

    from .vmec_tomnsp import TomnspsRZL

    if bool(host_update_assembly):
        fac_r_arr, fac_z_arr = _vmec_scale_m1_factors_from_mats_np(mats)
        if fac_r_arr.size == 0:
            return frzl_in
        ns_full = int(np.asarray(frzl_in.frcc).shape[0])
        nsolve = min(ns_full, int(fac_r_arr.shape[0]))
        if nsolve == ns_full:
            fac_r_full = fac_r_arr[:nsolve]
            fac_z_full = fac_z_arr[:nsolve]
        else:
            ones = np.ones((ns_full - nsolve,), dtype=fac_r_arr.dtype)
            fac_r_full = np.concatenate([fac_r_arr[:nsolve], ones])
            fac_z_full = np.concatenate([fac_z_arr[:nsolve], ones])
        frss = _scale_mode_slice_np(frzl_in.frss, mode_idx=1, scale=fac_r_full)
        fzcs = _scale_mode_slice_np(frzl_in.fzcs, mode_idx=1, scale=fac_z_full)
        frsc = _scale_mode_slice_np(getattr(frzl_in, "frsc", None), mode_idx=1, scale=fac_r_full)
        fzcc = _scale_mode_slice_np(getattr(frzl_in, "fzcc", None), mode_idx=1, scale=fac_z_full)
    else:
        fac_r_jax, fac_z_jax = _vmec_scale_m1_factors_from_mats(mats)
        if fac_r_jax.size == 0:
            return frzl_in
        fac_r = jnp.asarray(fac_r_jax, dtype=jnp.asarray(frzl_in.frcc).dtype)
        fac_z = jnp.asarray(fac_z_jax, dtype=jnp.asarray(frzl_in.fzsc).dtype)
        ns_full = int(jnp.asarray(frzl_in.frcc).shape[0])
        nsolve = min(ns_full, int(fac_r.shape[0]))
        ones_r = jnp.ones((max(ns_full - nsolve, 0),), dtype=jnp.asarray(frzl_in.frcc).dtype)
        ones_z = jnp.ones((max(ns_full - nsolve, 0),), dtype=jnp.asarray(frzl_in.fzsc).dtype)
        fac_r_full = fac_r[:nsolve] if nsolve == ns_full else jnp.concatenate([fac_r[:nsolve], ones_r], axis=0)
        fac_z_full = fac_z[:nsolve] if nsolve == ns_full else jnp.concatenate([fac_z[:nsolve], ones_z], axis=0)
        frss = _scale_mode_slice(frzl_in.frss, mode_idx=1, scale=fac_r_full)
        fzcs = _scale_mode_slice(frzl_in.fzcs, mode_idx=1, scale=fac_z_full)
        frsc = _scale_mode_slice(getattr(frzl_in, "frsc", None), mode_idx=1, scale=fac_r_full)
        fzcc = _scale_mode_slice(getattr(frzl_in, "fzcc", None), mode_idx=1, scale=fac_z_full)

    return TomnspsRZL(
        frcc=frzl_in.frcc,
        frss=frss,
        fzsc=frzl_in.fzsc,
        fzcs=fzcs,
        flsc=frzl_in.flsc,
        flcs=frzl_in.flcs,
        frsc=frsc,
        frcs=getattr(frzl_in, "frcs", None),
        fzcc=fzcc,
        fzss=getattr(frzl_in, "fzss", None),
        flcc=getattr(frzl_in, "flcc", None),
        flss=getattr(frzl_in, "flss", None),
    )


def _can_reassemble_precond_mats(mats: Any) -> bool:
    if not isinstance(mats, dict):
        return False
    required = (
        "arm_parity",
        "ard_parity",
        "brm_parity",
        "brd_parity",
        "azm_parity",
        "azd_parity",
        "bzm_parity",
        "bzd_parity",
        "cxd_full",
        "delta_s",
    )
    return all(key in mats for key in required)


def _maybe_dump_lam_gcl(
    *,
    frzl_pre,
    frzl_post,
    static,
    iter_idx: int,
    delta_s,
) -> None:
    env = os.getenv("VMEC_JAX_DUMP_LAM", "")
    if not env or env == "0":
        return
    iters = _parse_iter_list(os.getenv("VMEC_JAX_DUMP_LAM_ITER", ""))
    if iters is None:
        iters = _parse_iter_list(os.getenv("VMEC_JAX_DUMP_ITER", ""))
    if iters is not None and int(iter_idx) not in iters:
        return
    outdir = os.getenv("VMEC_JAX_DUMP_LAM_DIR", "")
    if not outdir:
        outdir = os.getenv("VMEC_JAX_DUMP_DIR", ".")
    outdir = Path(outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    ns = int(static.cfg.ns)
    path = outdir / f"lam_gcl_ns{ns}_iter{int(iter_idx)}.npz"

    _gcr_pre, _gcz_pre, gcl_pre = _gc_from_frzl(frzl=frzl_pre, cfg=static.cfg)
    _gcr_post, _gcz_post, gcl_post = _gc_from_frzl(frzl=frzl_post, cfg=static.cfg)

    gcl_pre = np.asarray(gcl_pre)
    gcl_post = np.asarray(gcl_post)
    delta_s_f = float(np.asarray(delta_s))
    # VMEC excludes the axis surface (js=1) from fsql1 sums.
    fsql1_pre = float(np.sum(gcl_pre[1:] * gcl_pre[1:]) * delta_s_f)
    fsql1_post = float(np.sum(gcl_post[1:] * gcl_post[1:]) * delta_s_f)

    _maybe_dump_lam_fsql1(
        fsql1_pre=fsql1_pre,
        fsql1_post=fsql1_post,
        static=static,
        iter_idx=int(iter_idx),
    )

    np.savez(
        path,
        gcl_pre=gcl_pre,
        gcl_post=gcl_post,
        fsql1_pre=fsql1_pre,
        fsql1_post=fsql1_post,
        delta_s=delta_s_f,
        ns=int(static.cfg.ns),
        mpol=int(static.cfg.mpol),
        ntor=int(static.cfg.ntor),
        lthreed=bool(static.cfg.lthreed),
        lasym=bool(static.cfg.lasym),
    )


_HLO_DUMPED_KEYS: set[tuple[str, int, int, int, int, int, bool]] = set()


def _maybe_dump_hlo_kernel(
    *,
    label: str,
    fn,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    static: Any,
    wout_like: Any,
    force: bool = False,
) -> None:
    env_dir = os.getenv("VMEC_JAX_DUMP_HLO_DIR", "").strip()
    if not env_dir:
        return
    env_all = os.getenv("VMEC_JAX_DUMP_HLO", "").strip().lower()
    enabled_all = env_all not in ("", "0", "false", "no")
    env_label = os.getenv(f"VMEC_JAX_DUMP_HLO_{label.upper()}", "").strip().lower()
    enabled_label = env_label not in ("", "0", "false", "no")
    if not force and not (enabled_all or enabled_label):
        return
    if not has_jax():
        return
    try:
        ns = int(getattr(static.cfg, "ns", 0))
        key = (
            str(label),
            ns,
            int(getattr(wout_like, "mpol", 0)),
            int(getattr(wout_like, "ntor", 0)),
            int(getattr(wout_like, "nfp", 0)),
            int(getattr(static.cfg, "ntheta", 0)),
            bool(getattr(wout_like, "lasym", False)),
        )
    except Exception:
        key = (str(label), 0, 0, 0, 0, 0, False)
    if key in _HLO_DUMPED_KEYS:
        return

    try:
        import jax
    except Exception:
        return

    outdir = Path(env_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    fname = f"hlo_{label}_ns{key[1]}_mpol{key[2]}_ntor{key[3]}.txt"
    outpath = outdir / fname

    hlo_text = None
    err_text = None
    try:
        jitted = jax.jit(fn)
        hlo = jitted.lower(*args, **kwargs).compiler_ir(dialect="hlo")
        if hasattr(hlo, "as_hlo_text"):
            hlo_text = hlo.as_hlo_text()
        elif hasattr(hlo, "as_text"):
            hlo_text = hlo.as_text()
        else:
            hlo_text = str(hlo)
    except Exception as exc:
        err_text = f"jit.lower failed: {exc!r}"
        try:
            hlo = jax.xla_computation(fn)(*args, **kwargs)
            if hasattr(hlo, "as_hlo_text"):
                hlo_text = hlo.as_hlo_text()
            else:
                hlo_text = str(hlo)
        except Exception as exc2:
            err_text = f"{err_text}\n xla_computation failed: {exc2!r}"
            hlo_text = None

    if hlo_text is None:
        if os.getenv("VMEC_JAX_DUMP_HLO_VERBOSE", "").strip().lower() not in ("", "0", "false", "no"):
            try:
                errpath = outdir / f"hlo_{label}_error_ns{key[1]}_mpol{key[2]}_ntor{key[3]}.txt"
                errpath.write_text(err_text or "unknown error")
            except Exception:
                pass
        return
    try:
        outpath.write_text(hlo_text)
        _HLO_DUMPED_KEYS.add(key)
    except Exception:
        return


def _maybe_dump_bsube(*, bc, static, iter_idx: int) -> None:
    env = os.getenv("VMEC_JAX_DUMP_BSUBE", "")
    if not env or env == "0":
        return
    iters = _parse_iter_list(os.getenv("VMEC_JAX_DUMP_ITER", ""))
    if iters is not None and int(iter_idx) not in iters:
        return
    outdir = Path(os.getenv("VMEC_JAX_DUMP_DIR", ".")).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    ns = int(static.cfg.ns)
    path = outdir / f"bsube_ns{ns}_iter{int(iter_idx)}.dat"

    bsubu = np.asarray(bc.bsubu_e_scaled)
    bsubv = np.asarray(bc.bsubv_e_scaled)
    ns, ntheta, nzeta = bsubu.shape

    with path.open("w", encoding="utf-8") as f:
        f.write("# bcovar bsube dump (scaled)\n")
        f.write(f"ns={ns}\n")
        f.write(f"ntheta3={ntheta}\n")
        f.write(f"nzeta={nzeta}\n")
        f.write(f"lamscale={float(np.asarray(bc.lamscale)):.16e}\n")
        f.write("columns: js lt lz bsubu_e bsubv_e\n")
        for lt in range(ntheta):
            for lz in range(nzeta):
                for js in range(ns):
                    f.write(f"{js + 1:6d}{lt + 1:6d}{lz + 1:6d}{bsubu[js, lt, lz]:24.16e}{bsubv[js, lt, lz]:24.16e}\n")


def _maybe_dump_bsube_terms(*, bc, static, iter_idx: int) -> None:
    env = os.getenv("VMEC_JAX_DUMP_BSUBE_TERMS", "")
    if not env or env == "0":
        return
    iters = _parse_iter_list(os.getenv("VMEC_JAX_DUMP_ITER", ""))
    if iters is not None and int(iter_idx) not in iters:
        return
    outdir = Path(os.getenv("VMEC_JAX_DUMP_DIR", ".")).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    ns = int(static.cfg.ns)
    path = outdir / f"bsube_terms_ns{ns}_iter{int(iter_idx)}.dat"

    lvv_sh = np.asarray(getattr(bc, "lvv_sh"))
    lu0 = np.asarray(getattr(bc, "lu0_force"))
    lu1 = np.asarray(getattr(bc, "lu1_full"))
    phip = np.asarray(getattr(bc, "phip_internal"))
    bsubu_tmp = np.asarray(getattr(bc, "bsubu_tmp"))
    bsubv_pre = np.asarray(getattr(bc, "bsubv_preblend"))

    ns, ntheta, nzeta = lvv_sh.shape
    with path.open("w", encoding="utf-8") as f:
        f.write("# bcovar bsube terms dump\n")
        f.write(f"ns={ns}\n")
        f.write(f"ntheta3={ntheta}\n")
        f.write(f"nzeta={nzeta}\n")
        f.write("columns: js lt lz lvv_sh lu0 lu1 phipf bsubu_tmp bsubv_pre\n")
        for lt in range(ntheta):
            for lz in range(nzeta):
                for js in range(ns):
                    f.write(
                        f"{js + 1:6d}{lt + 1:6d}{lz + 1:6d}"
                        f"{lvv_sh[js, lt, lz]:24.16e}{lu0[js, lt, lz]:24.16e}{lu1[js, lt, lz]:24.16e}"
                        f"{phip[js]:24.16e}{bsubu_tmp[js, lt, lz]:24.16e}{bsubv_pre[js, lt, lz]:24.16e}\n"
                    )


def _maybe_dump_bsubh(*, bc, static, iter_idx: int) -> None:
    env = os.getenv("VMEC_JAX_DUMP_BSUBH", "")
    if not env or env == "0":
        return
    iters = _parse_iter_list(os.getenv("VMEC_JAX_DUMP_ITER", ""))
    if iters is not None and int(iter_idx) not in iters:
        return
    outdir = Path(os.getenv("VMEC_JAX_DUMP_DIR", ".")).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    ns = int(static.cfg.ns)
    path = outdir / f"bsubh_ns{ns}_iter{int(iter_idx)}.dat"

    bsubu = np.asarray(getattr(bc, "bsubu"))
    bsubv = np.asarray(getattr(bc, "bsubv"))

    ns, ntheta, nzeta = bsubu.shape
    with path.open("w", encoding="utf-8") as f:
        f.write("# bcovar bsubh dump (half mesh)\n")
        f.write(f"ns={ns}\n")
        f.write(f"ntheta3={ntheta}\n")
        f.write(f"nzeta={nzeta}\n")
        f.write("columns: js lt lz bsubuh bsubvh\n")
        for lt in range(ntheta):
            for lz in range(nzeta):
                for js in range(ns):
                    f.write(f"{js + 1:6d}{lt + 1:6d}{lz + 1:6d}{bsubu[js, lt, lz]:24.16e}{bsubv[js, lt, lz]:24.16e}\n")


def _maybe_dump_bsubs(*, bc, state, static, trig, iter_idx: int, kernels=None) -> None:
    env = os.getenv("VMEC_JAX_DUMP_BSUBS", "")
    if not env or env == "0":
        return
    iters = _parse_iter_list(os.getenv("VMEC_JAX_DUMP_ITER", ""))
    if iters is not None and int(iter_idx) not in iters:
        return
    outdir = Path(os.getenv("VMEC_JAX_DUMP_DIR", ".")).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    ns = int(static.cfg.ns)
    path = outdir / f"bsubs_ns{ns}_iter{int(iter_idx)}.npz"

    from .wout import _compute_bsubs_half_mesh, _vmec_symforce_apply

    s = np.asarray(static.s, dtype=float)
    bsupu = np.asarray(bc.bsupu)
    bsupv = np.asarray(bc.bsupv)
    force_rs = None
    force_zs = None
    force_ru12 = None
    force_zu12 = None
    _force_bss_env = os.getenv("VMEC_JAX_WOUT_FORCE_BSS", "").strip().lower()
    if _force_bss_env == "":
        use_force_bss = not bool(static.cfg.lasym)
    else:
        use_force_bss = _force_bss_env not in ("0", "false", "no")

    def _force_sym(arr, kind: str):
        arr_np = np.asarray(arr, dtype=float)
        if not bool(static.cfg.lasym):
            return arr_np
        return _vmec_symforce_apply(f=arr_np, trig=trig, kind=kind)

    if use_force_bss and kernels is not None:
        if hasattr(kernels, "crmn_e"):
            bsupu = _force_sym(getattr(kernels, "crmn_e"), "crs")
        if hasattr(kernels, "czmn_e"):
            bsupv = _force_sym(getattr(kernels, "czmn_e"), "czs")
        if hasattr(kernels, "bzmn_e"):
            force_rs = _force_sym(getattr(kernels, "bzmn_e"), "bzs")
        if hasattr(kernels, "brmn_e"):
            force_zs = _force_sym(getattr(kernels, "brmn_e"), "brs")
        if hasattr(kernels, "azmn_e"):
            force_ru12 = _force_sym(getattr(kernels, "azmn_e"), "azs")
        if hasattr(kernels, "armn_e"):
            force_zu12 = _force_sym(getattr(kernels, "armn_e"), "ars")
    bsubu = np.asarray(bc.bsubu)
    bsubv = np.asarray(bc.bsubv)
    sqrtg = np.asarray(bc.jac.sqrtg)

    geom_terms = {}
    if kernels is not None:
        for name in (
            "pr1_even",
            "pr1_odd",
            "pz1_even",
            "pz1_odd",
            "pru_even",
            "pru_odd",
            "pzu_even",
            "pzu_odd",
            "prv_even",
            "prv_odd",
            "pzv_even",
            "pzv_odd",
        ):
            if hasattr(kernels, name):
                geom_terms[name] = np.asarray(getattr(kernels, name), dtype=float)

    bsubs_half = _compute_bsubs_half_mesh(
        state=state,
        geom_modes=static.modes,
        s=s,
        lconm1=bool(getattr(static.cfg, "lconm1", True)),
        lthreed=bool(static.cfg.ntor > 0),
        lasym=bool(static.cfg.lasym),
        bsupu=bsupu,
        bsupv=bsupv,
        trig=trig,
        geom=geom_terms,
        jac_half=bc.jac,
        force_rs=force_rs,
        force_zs=force_zs,
        force_ru12=force_ru12,
        force_zu12=force_zu12,
    )
    bsubs_full = np.asarray(bsubs_half, dtype=float).copy()
    if ns > 2:
        bsubs_full[1:-1] = 0.5 * (bsubs_full[1:-1] + bsubs_full[2:])
    if ns > 0:
        bsubs_full[0] = 0.0
        bsubs_full[-1] = 0.0

    # JXBFORCE-style full-mesh bsupu/bsupv averages (for comparison with jxbout).
    bsupu1 = np.zeros_like(bsupu)
    bsupv1 = np.zeros_like(bsupv)
    if ns > 1:
        sqrtg_half = 0.5 * (sqrtg[1:] + sqrtg[:-1])
        denom = np.where(sqrtg_half != 0.0, sqrtg_half, 1.0)
        if ns > 2:
            # VMEC jxbforce: bsupu1(js) = 0.5*(bsupu(js)*gsqrt(js) + bsupu(js+1)*gsqrt(js+1)) / sqrtg_half
            bsupu1[1:-1] = 0.5 * (bsupu[1:-1] * sqrtg[1:-1] + bsupu[2:] * sqrtg[2:]) / denom[1:]
            bsupv1[1:-1] = 0.5 * (bsupv[1:-1] * sqrtg[1:-1] + bsupv[2:] * sqrtg[2:]) / denom[1:]
        bsupu1[0] = 0.0
        bsupu1[-1] = 0.0
        bsupv1[0] = 0.0
        bsupv1[-1] = 0.0

    np.savez(
        path,
        bsubs_half=np.asarray(bsubs_half, dtype=float),
        bsubs_full=np.asarray(bsubs_full, dtype=float),
        bsupu=np.asarray(bsupu, dtype=float),
        bsupv=np.asarray(bsupv, dtype=float),
        bsupu1=np.asarray(bsupu1, dtype=float),
        bsupv1=np.asarray(bsupv1, dtype=float),
        bsubu=np.asarray(bsubu, dtype=float),
        bsubv=np.asarray(bsubv, dtype=float),
        sqrtg=np.asarray(sqrtg, dtype=float),
        s=np.asarray(s, dtype=float),
    )


def _maybe_dump_lulv(
    *,
    bc,
    static,
    iter_idx: int,
    state: VMECState | None = None,
    trig: VmecTrigTables | None = None,
) -> None:
    env = os.getenv("VMEC_JAX_DUMP_LULV", "")
    if not env or env == "0":
        return
    iters = _parse_iter_list(os.getenv("VMEC_JAX_DUMP_ITER", ""))
    if iters is not None and int(iter_idx) not in iters:
        return
    outdir = Path(os.getenv("VMEC_JAX_DUMP_DIR", ".")).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    ns = int(static.cfg.ns)
    path = outdir / f"lulv_ns{ns}_iter{int(iter_idx)}.npz"
    data = {
        "lu0_full": np.asarray(getattr(bc, "lu0_full")),
        "lu1_full": np.asarray(getattr(bc, "lu1_full")),
        "lv0_full": np.asarray(getattr(bc, "lv0_full")),
        "lv1_full": np.asarray(getattr(bc, "lv1_full")),
    }
    if state is not None:
        data["Lcos"] = np.asarray(state.Lcos)
        data["Lsin"] = np.asarray(state.Lsin)
        data["m_modes"] = np.asarray(static.modes.m, dtype=int)
        data["n_modes"] = np.asarray(static.modes.n, dtype=int)
        if trig is not None:
            # Debug lambda odd-m synthesis inputs (physical odd pieces).
            from .vmec_realspace import vmec_realspace_synthesis_dtheta, vmec_realspace_synthesis_dzeta_phys

            m_modes = np.asarray(static.modes.m, dtype=int)
            mask_m1 = (m_modes == 1).astype(np.asarray(state.Lsin).dtype)
            mask_odd_rest = ((m_modes % 2 == 1) & (m_modes != 1)).astype(np.asarray(state.Lsin).dtype)
            lu_m1 = vmec_realspace_synthesis_dtheta(
                coeff_cos=jnp.asarray(state.Lcos) * mask_m1,
                coeff_sin=jnp.asarray(state.Lsin) * mask_m1,
                modes=static.modes,
                trig=trig,
                coeffs_internal=True,
                apply_scalxc=True,
                s=static.s,
            )
            lu_rest = vmec_realspace_synthesis_dtheta(
                coeff_cos=jnp.asarray(state.Lcos) * mask_odd_rest,
                coeff_sin=jnp.asarray(state.Lsin) * mask_odd_rest,
                modes=static.modes,
                trig=trig,
                coeffs_internal=True,
                apply_scalxc=True,
                s=static.s,
            )
            lv_m1 = vmec_realspace_synthesis_dzeta_phys(
                coeff_cos=jnp.asarray(state.Lcos) * mask_m1,
                coeff_sin=jnp.asarray(state.Lsin) * mask_m1,
                modes=static.modes,
                trig=trig,
                coeffs_internal=True,
                apply_scalxc=True,
                s=static.s,
            )
            lv_rest = vmec_realspace_synthesis_dzeta_phys(
                coeff_cos=jnp.asarray(state.Lcos) * mask_odd_rest,
                coeff_sin=jnp.asarray(state.Lsin) * mask_odd_rest,
                modes=static.modes,
                trig=trig,
                coeffs_internal=True,
                apply_scalxc=True,
                s=static.s,
            )
            data["lu_phys_m1"] = np.asarray(lu_m1)
            data["lu_phys_rest"] = np.asarray(lu_rest)
            data["lv_phys_m1"] = np.asarray(lv_m1)
            data["lv_phys_rest"] = np.asarray(lv_rest)
    np.savez(path, **data)


def _maybe_dump_precond_inputs(*, bc, trig, static, iter_idx: int, kernels=None) -> None:
    env = os.getenv("VMEC_JAX_DUMP_PRECOND", "")
    if not env or env == "0":
        return
    iters = _parse_iter_list(os.getenv("VMEC_JAX_DUMP_ITER", ""))
    if iters is not None and int(iter_idx) not in iters:
        return
    outdir = Path(os.getenv("VMEC_JAX_DUMP_DIR", ".")).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"precond_inputs_iter{int(iter_idx)}.dat"

    try:
        r12 = np.asarray(bc.jac.r12)
        sqrtg = np.asarray(bc.jac.sqrtg)
        bsq = np.asarray(bc.bsq)
        ru12 = np.asarray(bc.jac.ru12)
        zu12 = np.asarray(bc.jac.zu12)
    except Exception:
        return

    wint3 = getattr(trig, "wint3_precond", None) if trig is not None else None
    if wint3 is None:
        # Fallback to uniform weights if trig tables are missing.
        wint3 = np.ones((1, r12.shape[1], r12.shape[2]), dtype=float)
    wint3 = np.asarray(wint3)
    if wint3.ndim != 3:
        return
    if wint3.shape[0] == 1:
        wint_full = np.broadcast_to(wint3, r12.shape)
    elif wint3.shape[0] == r12.shape[0]:
        wint_full = wint3
    else:
        wint_full = np.broadcast_to(wint3[:1, :, :], r12.shape)

    ns = int(r12.shape[0])
    ntheta3 = int(r12.shape[1])
    nzeta = int(r12.shape[2])

    with path.open("w", encoding="utf-8") as f:
        f.write("# precond inputs (vmec_jax)\n")
        f.write(f"ns={ns}\n")
        f.write(f"nzeta={nzeta}\n")
        f.write(f"ntheta3={ntheta3}\n")
        f.write("columns: js lt lz\n")
        f.write("         r12 sqrtg bsq\n")
        f.write("         ru12 zu12 wint\n")
        for lt in range(ntheta3):
            for lz in range(nzeta):
                for j in range(1, ns):
                    f.write(
                        f"{j + 1:6d}{lt + 1:6d}{lz + 1:6d}"
                        f"{r12[j, lt, lz]:24.16E}"
                        f"{sqrtg[j, lt, lz]:24.16E}"
                        f"{bsq[j, lt, lz]:24.16E}"
                        f"{ru12[j, lt, lz]:24.16E}"
                        f"{zu12[j, lt, lz]:24.16E}"
                        f"{wint_full[j, lt, lz]:24.16E}\n"
                    )

    if kernels is None:
        return
    try:
        hidden = {
            "tau": np.asarray(bc.jac.tau),
            "rs": np.asarray(bc.jac.rs),
            "zs": np.asarray(bc.jac.zs),
            "pru_even": np.asarray(kernels.pru_even),
            "pru_odd": np.asarray(kernels.pru_odd),
            "pzu_even": np.asarray(kernels.pzu_even),
            "pzu_odd": np.asarray(kernels.pzu_odd),
            "pr1_odd": np.asarray(kernels.pr1_odd),
            "pz1_odd": np.asarray(kernels.pz1_odd),
        }
    except Exception:
        return
    np.savez(outdir / f"precond_hidden_iter{int(iter_idx)}.npz", **hidden)


def _maybe_dump_gmetric(*, bc, static, iter_idx: int) -> None:
    env = os.getenv("VMEC_JAX_DUMP_GMETRIC", "")
    if not env or env == "0":
        return
    iters = _parse_iter_list(os.getenv("VMEC_JAX_DUMP_ITER", ""))
    if iters is not None and int(iter_idx) not in iters:
        return
    outdir = Path(os.getenv("VMEC_JAX_DUMP_DIR", ".")).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"gmetric_iter{int(iter_idx)}.dat"

    try:
        guu = np.asarray(bc.guu, dtype=float)
        guv = np.asarray(bc.guv, dtype=float)
        gvv = np.asarray(bc.gvv, dtype=float)
    except Exception:
        return

    if guu.ndim != 3 or guv.shape != guu.shape or gvv.shape != guu.shape:
        return

    # VMEC dumps `gmetric_iter*.dat` before the cylindrical `R^2` term is
    # added into `pgvv`, while the live JAX half-mesh fields store the later
    # post-`R^2` metric needed by `bsubv`, `wb`, and `wout` parity.
    gmetric_guu = np.array(guu, copy=True)
    gmetric_guv = np.array(guv, copy=True)
    gmetric_gvv = np.array(gvv, copy=True)
    try:
        r12 = np.asarray(bc.jac.r12, dtype=float)
        if r12.shape == gmetric_gvv.shape:
            gmetric_gvv = gmetric_gvv - (r12 * r12)
    except Exception:
        pass
    if gmetric_guu.shape[0] >= 1:
        gmetric_guu[0, :, :] = 0.0
        gmetric_guv[0, :, :] = 0.0
        gmetric_gvv[0, :, :] = 0.0

    ns = int(guu.shape[0])
    ntheta3 = int(guu.shape[1])
    nzeta = int(guu.shape[2])

    with path.open("w", encoding="utf-8") as f:
        f.write("# bcovar metric dump (half mesh)\n")
        f.write(f"ns={ns}\n")
        f.write(f"ntheta3={ntheta3}\n")
        f.write(f"nzeta={nzeta}\n")
        f.write("columns: js lt lz pguu pguv pgvv\n")
        for lt in range(ntheta3):
            for lz in range(nzeta):
                for js in range(ns):
                    f.write(
                        f"{js + 1:6d}{lt + 1:6d}{lz + 1:6d}"
                        f"{gmetric_guu[js, lt, lz]:24.16e}"
                        f"{gmetric_guv[js, lt, lz]:24.16e}"
                        f"{gmetric_gvv[js, lt, lz]:24.16e}\n"
                    )


def _maybe_dump_xc(
    *,
    state: VMECState,
    vRcc,
    vRss,
    vZsc,
    vZcs,
    vLsc,
    vLcs,
    vRsc=None,
    vRcs=None,
    vZcc=None,
    vZss=None,
    vLcc=None,
    vLss=None,
    static,
    iter_idx: int,
) -> None:
    env = os.getenv("VMEC_JAX_DUMP_XC", "")
    if not env or env == "0":
        return
    iters = _parse_iter_list(os.getenv("VMEC_JAX_DUMP_ITER", ""))
    if iters is not None and int(iter_idx) not in iters:
        return
    outdir = Path(os.getenv("VMEC_JAX_DUMP_DIR", ".")).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    ns_val = int(static.cfg.ns)
    path = outdir / f"xc_ns{ns_val}_iter{int(iter_idx)}.npz"
    from .diagnostics import vmec_internal_mn_from_state, vmec_xc_from_mn_blocks

    blocks = vmec_internal_mn_from_state(state, static, apply_basis_norm=False, apply_m1_constraint=False)
    xc_kwargs = dict(
        rcc=blocks["rcc"],
        rss=blocks["rss"],
        zsc=blocks["zsc"],
        zcs=blocks["zcs"],
        lsc=blocks["lsc"],
        lcs=blocks["lcs"],
    )
    if "rsc" in blocks:
        xc_kwargs.update(
            rsc=blocks.get("rsc"),
            rcs=blocks.get("rcs"),
            zcc=blocks.get("zcc"),
            zss=blocks.get("zss"),
            lcc=blocks.get("lcc"),
            lss=blocks.get("lss"),
        )
    xc = vmec_xc_from_mn_blocks(cfg=static.cfg, **xc_kwargs)

    xcdot_kwargs = dict(
        rcc=np.asarray(vRcc),
        rss=np.asarray(vRss),
        zsc=np.asarray(vZsc),
        zcs=np.asarray(vZcs),
        lsc=np.asarray(vLsc),
        lcs=np.asarray(vLcs),
    )
    # Include asymmetric channels when present so LASYM xc/v dumps match VMEC's
    # internal stacking and do not report false zero-components.
    if vRsc is not None:
        xcdot_kwargs["rsc"] = np.asarray(vRsc)
    if vRcs is not None:
        xcdot_kwargs["rcs"] = np.asarray(vRcs)
    if vZcc is not None:
        xcdot_kwargs["zcc"] = np.asarray(vZcc)
    if vZss is not None:
        xcdot_kwargs["zss"] = np.asarray(vZss)
    if vLcc is not None:
        xcdot_kwargs["lcc"] = np.asarray(vLcc)
    if vLss is not None:
        xcdot_kwargs["lss"] = np.asarray(vLss)
    xcdot = vmec_xc_from_mn_blocks(cfg=static.cfg, **xcdot_kwargs)
    np.savez(
        path,
        xc=np.asarray(xc),
        xcdot=np.asarray(xcdot),
        v=np.asarray(xcdot),
        ns=int(static.cfg.ns),
        mpol=int(static.cfg.mpol),
        ntor=int(static.cfg.ntor),
        lthreed=bool(static.cfg.lthreed),
        lasym=bool(static.cfg.lasym),
    )


def _mode00_index(modes) -> Optional[int]:
    m = np.asarray(modes.m)
    n = np.asarray(modes.n)
    idx = np.where((m == 0) & (n == 0))[0]
    if idx.size == 0:
        return None
    return int(idx[0])


def _enforce_lambda_gauge(Lcos, Lsin, *, idx00: Optional[int]):
    """Fix the (m,n)=(0,0) gauge mode to 0 (it is a nullspace)."""
    if idx00 is None:
        return Lcos, Lsin
    Lcos = _zero_coeff_column(Lcos, idx=int(idx00))
    Lsin = _zero_coeff_column(Lsin, idx=int(idx00))
    return Lcos, Lsin


def _apply_vmec_lambda_axis_rules_to_state(
    st: VMECState,
    *,
    enforce_vmec_lambda_axis: bool,
    host_update_assembly: bool,
    idx00: Optional[int],
) -> VMECState:
    """Enforce the VMEC lambda gauge while preserving stored axis coefficients."""
    if not bool(enforce_vmec_lambda_axis):
        return st
    if bool(host_update_assembly):
        Lcos = np.array(np.asarray(st.Lcos))
        Lsin = np.array(np.asarray(st.Lsin))
        if idx00 is not None:
            ncols = Lcos.shape[1]
            if 0 <= int(idx00) < ncols:
                Lcos[:, int(idx00)] = 0.0
                Lsin[:, int(idx00)] = 0.0
        return VMECState(
            layout=st.layout,
            Rcos=st.Rcos,
            Rsin=st.Rsin,
            Zcos=st.Zcos,
            Zsin=st.Zsin,
            Lcos=Lcos,
            Lsin=Lsin,
        )
    Lcos = jnp.asarray(st.Lcos)
    Lsin = jnp.asarray(st.Lsin)
    Lcos, Lsin = _enforce_lambda_gauge(Lcos, Lsin, idx00=idx00)
    return VMECState(
        layout=st.layout,
        Rcos=st.Rcos,
        Rsin=st.Rsin,
        Zcos=st.Zcos,
        Zsin=st.Zsin,
        Lcos=Lcos,
        Lsin=Lsin,
    )


def _axis_m0_mask(static, *, dtype):
    if getattr(static, "m_is_m0", None) is not None:
        return jnp.asarray(static.m_is_m0, dtype=dtype)
    m = jnp.asarray(static.modes.m)
    return (m == 0).astype(dtype)


def _zero_coeff_column(arr, *, idx: int):
    """Zero one Fourier coefficient column with concatenation instead of masking."""
    arr = jnp.asarray(arr)
    ncols = int(arr.shape[1])
    if idx < 0 or idx >= ncols:
        return arr
    zero = jnp.zeros_like(arr[:, :1])
    if idx == 0:
        if ncols == 1:
            return zero
        return jnp.concatenate([zero, arr[:, 1:]], axis=1)
    if idx == ncols - 1:
        return jnp.concatenate([arr[:, :idx], zero], axis=1)
    return jnp.concatenate([arr[:, :idx], zero, arr[:, idx + 1 :]], axis=1)


def _replace_mode_slice(arr, *, mode_idx: int, replacement):
    """Replace one `(m, :)` slice of a `(ns, mpol, nrange)` array."""
    if arr is None:
        return None
    arr = jnp.asarray(arr)
    nmodes = int(arr.shape[1])
    if mode_idx < 0 or mode_idx >= nmodes:
        return arr
    repl = jnp.asarray(replacement, dtype=arr.dtype)[:, None, :]
    if mode_idx == 0:
        if nmodes == 1:
            return repl
        return jnp.concatenate([repl, arr[:, 1:, :]], axis=1)
    if mode_idx == nmodes - 1:
        return jnp.concatenate([arr[:, :mode_idx, :], repl], axis=1)
    return jnp.concatenate([arr[:, :mode_idx, :], repl, arr[:, mode_idx + 1 :, :]], axis=1)


def _scale_mode_slice(arr, *, mode_idx: int, scale):
    """Scale one `(m, :)` slice of a `(ns, mpol, nrange)` array."""
    if arr is None:
        return None
    arr = jnp.asarray(arr)
    nmodes = int(arr.shape[1])
    if mode_idx < 0 or mode_idx >= nmodes:
        return arr
    scaled = arr[:, mode_idx, :] * jnp.asarray(scale, dtype=arr.dtype)[:, None]
    return _replace_mode_slice(arr, mode_idx=mode_idx, replacement=scaled)


def _zero_coeff_column_np(arr, *, idx: int) -> np.ndarray:
    """NumPy in-place version of _zero_coeff_column (no jnp.concatenate)."""
    arr = np.array(np.asarray(arr))
    ncols = int(arr.shape[1])
    if 0 <= idx < ncols:
        arr[:, idx] = 0.0
    return arr


def _replace_mode_slice_np(arr, *, mode_idx: int, replacement):
    """NumPy in-place version of _replace_mode_slice."""
    if arr is None:
        return None
    arr = np.array(np.asarray(arr))
    nmodes = int(arr.shape[1])
    if 0 <= mode_idx < nmodes:
        arr[:, mode_idx, :] = np.asarray(replacement)
    return arr


def _scale_mode_slice_np(arr, *, mode_idx: int, scale):
    """NumPy in-place version of _scale_mode_slice."""
    if arr is None:
        return None
    arr = np.array(np.asarray(arr))
    nmodes = int(arr.shape[1])
    if 0 <= mode_idx < nmodes:
        arr[:, mode_idx, :] *= np.asarray(scale)[:, None]
    return arr


def _enforce_field_rows(arr, *, axis_mask=None, edge_row=None, zero_axis: bool = False):
    """Apply axis/edge row constraints with at most one concatenation."""
    arr = jnp.asarray(arr)
    ns = int(arr.shape[0])
    if ns == 0:
        return arr

    first = arr[:1, :]
    if zero_axis:
        first = jnp.zeros_like(first)
    elif axis_mask is not None:
        first = first * jnp.asarray(axis_mask, dtype=arr.dtype)[None, :]

    last = arr[-1:, :]
    if edge_row is not None:
        last = jnp.asarray(edge_row, dtype=arr.dtype)[None, :]

    if ns == 1:
        row = last
        if zero_axis:
            row = jnp.zeros_like(row)
        elif axis_mask is not None:
            row = row * jnp.asarray(axis_mask, dtype=arr.dtype)[None, :]
        return row

    if (zero_axis or axis_mask is not None) and (edge_row is not None):
        return jnp.concatenate([first, arr[1:-1, :], last], axis=0)
    if zero_axis or axis_mask is not None:
        return jnp.concatenate([first, arr[1:, :]], axis=0)
    if edge_row is not None:
        return jnp.concatenate([arr[:-1, :], last], axis=0)
    return arr


def _enforce_fixed_boundary_and_axis(
    state: VMECState,
    static,
    *,
    edge_Rcos,
    edge_Rsin,
    edge_Zcos,
    edge_Zsin,
    enforce_axis: bool = True,
    enforce_edge: bool = True,
    enforce_lambda_axis: bool = True,
    idx00: Optional[int],
) -> VMECState:
    """Apply minimal VMEC regularity + fixed-boundary constraints.

    - Fix R/Z at the outer surface (s=1) to preserve the prescribed boundary.
    - Enforce axis regularity by zeroing all m>0 Fourier coefficients at s=0.
    - Enforce lambda gauge (m,n)=(0,0) = 0 everywhere.
    """
    Rcos = jnp.asarray(state.Rcos)
    Rsin = jnp.asarray(state.Rsin)
    Zcos = jnp.asarray(state.Zcos)
    Zsin = jnp.asarray(state.Zsin)
    Lcos = jnp.asarray(state.Lcos)
    Lsin = jnp.asarray(state.Lsin)

    mask_m0 = _axis_m0_mask(static, dtype=Rcos.dtype) if enforce_axis else None
    edge_Rcos_arr = edge_Rcos if enforce_edge else None
    edge_Rsin_arr = edge_Rsin if enforce_edge else None
    edge_Zcos_arr = edge_Zcos if enforce_edge else None
    edge_Zsin_arr = edge_Zsin if enforce_edge else None

    Rcos = _enforce_field_rows(Rcos, axis_mask=mask_m0, edge_row=edge_Rcos_arr)
    Rsin = _enforce_field_rows(Rsin, axis_mask=mask_m0, edge_row=edge_Rsin_arr)
    Zcos = _enforce_field_rows(Zcos, axis_mask=mask_m0, edge_row=edge_Zcos_arr)
    Zsin = _enforce_field_rows(Zsin, axis_mask=mask_m0, edge_row=edge_Zsin_arr)
    Lcos = _enforce_field_rows(Lcos, zero_axis=bool(enforce_lambda_axis))
    Lsin = _enforce_field_rows(Lsin, zero_axis=bool(enforce_lambda_axis))

    Lcos, Lsin = _enforce_lambda_gauge(Lcos, Lsin, idx00=idx00)

    return VMECState(
        layout=state.layout,
        Rcos=Rcos,
        Rsin=Rsin,
        Zcos=Zcos,
        Zsin=Zsin,
        Lcos=Lcos,
        Lsin=Lsin,
    )


def _enforce_field_rows_np(arr, *, axis_mask=None, edge_row=None, zero_axis: bool = False):
    """NumPy in-place version of _enforce_field_rows (avoids jnp.concatenate overhead)."""
    arr = np.array(arr)  # writable copy
    ns = arr.shape[0]
    if ns == 0:
        return arr
    if ns == 1:
        if edge_row is not None:
            arr[0] = np.asarray(edge_row)
        if zero_axis:
            arr[0] = 0.0
        elif axis_mask is not None:
            arr[0] *= np.asarray(axis_mask)
        return arr
    if edge_row is not None:
        arr[-1] = np.asarray(edge_row)
    if zero_axis:
        arr[0] = 0.0
    elif axis_mask is not None:
        arr[0] *= np.asarray(axis_mask)
    return arr


def _enforce_fixed_boundary_and_axis_np(
    state: VMECState,
    static,
    *,
    edge_Rcos,
    edge_Rsin,
    edge_Zcos,
    edge_Zsin,
    enforce_axis: bool = True,
    enforce_edge: bool = True,
    enforce_lambda_axis: bool = True,
    idx00: Optional[int],
    precomputed_axis_mask: Optional[np.ndarray] = None,
) -> VMECState:
    """NumPy version of _enforce_fixed_boundary_and_axis.

    Avoids 8 jnp.concatenate calls by using in-place NumPy row assignment.
    Only safe to call from the non-scan CPU path (host_update_assembly=True).
    Pass `precomputed_axis_mask` to avoid recomputing _axis_m0_mask every call.
    """
    Rcos = np.array(state.Rcos)
    Rsin = np.array(state.Rsin)
    Zcos = np.array(state.Zcos)
    Zsin = np.array(state.Zsin)
    Lcos = np.array(state.Lcos)
    Lsin = np.array(state.Lsin)

    if precomputed_axis_mask is not None:
        mask_m0 = precomputed_axis_mask if enforce_axis else None
    else:
        mask_m0 = np.asarray(_axis_m0_mask(static, dtype=Rcos.dtype)) if enforce_axis else None
    edge_Rcos_np = np.asarray(edge_Rcos) if enforce_edge else None
    edge_Rsin_np = np.asarray(edge_Rsin) if enforce_edge else None
    edge_Zcos_np = np.asarray(edge_Zcos) if enforce_edge else None
    edge_Zsin_np = np.asarray(edge_Zsin) if enforce_edge else None

    Rcos = _enforce_field_rows_np(Rcos, axis_mask=mask_m0, edge_row=edge_Rcos_np)
    Rsin = _enforce_field_rows_np(Rsin, axis_mask=mask_m0, edge_row=edge_Rsin_np)
    Zcos = _enforce_field_rows_np(Zcos, axis_mask=mask_m0, edge_row=edge_Zcos_np)
    Zsin = _enforce_field_rows_np(Zsin, axis_mask=mask_m0, edge_row=edge_Zsin_np)
    Lcos = _enforce_field_rows_np(Lcos, zero_axis=bool(enforce_lambda_axis))
    Lsin = _enforce_field_rows_np(Lsin, zero_axis=bool(enforce_lambda_axis))

    # Enforce lambda (m,n)=(0,0) gauge: zero that coefficient column in-place.
    if idx00 is not None and 0 <= int(idx00) < Lcos.shape[1]:
        Lcos[:, int(idx00)] = 0.0
        Lsin[:, int(idx00)] = 0.0

    # Return NumPy VMECState — JAX JIT converts at call site, avoiding 6 eager dispatches.
    return VMECState(
        layout=state.layout,
        Rcos=Rcos,
        Rsin=Rsin,
        Zcos=Zcos,
        Zsin=Zsin,
        Lcos=Lcos,
        Lsin=Lsin,
    )


def _grad_rms_state(grad: VMECState) -> float:
    g = np.asarray(grad.Rcos) ** 2
    g = g + np.asarray(grad.Rsin) ** 2
    g = g + np.asarray(grad.Zcos) ** 2
    g = g + np.asarray(grad.Zsin) ** 2
    g = g + np.asarray(grad.Lcos) ** 2
    g = g + np.asarray(grad.Lsin) ** 2
    return float(np.sqrt(np.mean(g)))


def _dtype_eps(dtype: Any) -> float:
    return float(np.finfo(np.dtype(dtype)).eps)


def _dtype_tiny(dtype: Any) -> float:
    return float(np.finfo(np.dtype(dtype)).tiny)


def _resolve_grad_tol(
    grad_tol: float | None,
    *,
    grad_rms0: float,
    dtype: Any,
) -> float:
    if grad_tol is not None:
        grad_tol = float(grad_tol)
        if grad_tol < 0.0:
            raise ValueError("grad_tol must be >= 0")
        return grad_tol
    scale = max(abs(float(grad_rms0)), _dtype_tiny(dtype))
    return float(np.sqrt(_dtype_eps(dtype)) * scale)


def _resolve_cg_tol(
    cg_tol: float | None,
    *,
    current_obj: float,
    initial_obj: float,
    target_obj: float,
    dtype: Any,
) -> float:
    if cg_tol is not None:
        cg_tol = float(cg_tol)
        if cg_tol <= 0.0:
            raise ValueError("cg_tol must be > 0")
        return cg_tol
    tiny = _dtype_tiny(dtype)
    denom = max(abs(float(initial_obj)), abs(float(target_obj)), tiny)
    ratio = max(abs(float(current_obj)), tiny) / denom
    eta = ratio / (1.0 + ratio)
    return float(max(eta, np.sqrt(_dtype_eps(dtype))))


def _resolve_lm_damping(
    damping: float | None,
    *,
    curvature_scale: float,
    dtype: Any,
) -> float:
    if damping is not None:
        damping = float(damping)
        if damping < 0.0:
            raise ValueError("damping must be nonnegative")
        return damping
    return float(np.sqrt(_dtype_eps(dtype)) * max(abs(float(curvature_scale)), _dtype_tiny(dtype)))


def _update_state_gd(state: VMECState, grad: VMECState, *, step: float, scale_rz: float, scale_l: float) -> VMECState:
    step = jnp.asarray(step, dtype=jnp.asarray(state.Rcos).dtype)
    scale_rz = jnp.asarray(scale_rz, dtype=step.dtype)
    scale_l = jnp.asarray(scale_l, dtype=step.dtype)
    return VMECState(
        layout=state.layout,
        Rcos=jnp.asarray(state.Rcos) - step * scale_rz * jnp.asarray(grad.Rcos),
        Rsin=jnp.asarray(state.Rsin) - step * scale_rz * jnp.asarray(grad.Rsin),
        Zcos=jnp.asarray(state.Zcos) - step * scale_rz * jnp.asarray(grad.Zcos),
        Zsin=jnp.asarray(state.Zsin) - step * scale_rz * jnp.asarray(grad.Zsin),
        Lcos=jnp.asarray(state.Lcos) - step * scale_l * jnp.asarray(grad.Lcos),
        Lsin=jnp.asarray(state.Lsin) - step * scale_l * jnp.asarray(grad.Lsin),
    )


def _mask_grad_for_constraints(
    grad: VMECState,
    static,
    *,
    idx00: Optional[int],
    mask_lambda_axis: bool = True,
) -> VMECState:
    """Project gradients onto the feasible set implied by our constraints."""
    gRcos = jnp.asarray(grad.Rcos)
    gRsin = jnp.asarray(grad.Rsin)
    gZcos = jnp.asarray(grad.Zcos)
    gZsin = jnp.asarray(grad.Zsin)
    gLcos = jnp.asarray(grad.Lcos)
    gLsin = jnp.asarray(grad.Lsin)

    # Fixed-boundary: don't update the edge surface for R/Z.
    gRcos = gRcos.at[-1, :].set(0.0)
    gRsin = gRsin.at[-1, :].set(0.0)
    gZcos = gZcos.at[-1, :].set(0.0)
    gZsin = gZsin.at[-1, :].set(0.0)

    # Axis regularity: don't update m>0 coefficients at s=0 for R/Z.
    m = jnp.asarray(static.modes.m)
    mask_m0 = (m == 0).astype(gRcos.dtype)
    gRcos = gRcos.at[0, :].set(gRcos[0, :] * mask_m0)
    gRsin = gRsin.at[0, :].set(gRsin[0, :] * mask_m0)
    gZcos = gZcos.at[0, :].set(gZcos[0, :] * mask_m0)
    gZsin = gZsin.at[0, :].set(gZsin[0, :] * mask_m0)

    # Lambda: optionally fix the axis row.
    if bool(mask_lambda_axis):
        gLcos = gLcos.at[0, :].set(0.0)
        gLsin = gLsin.at[0, :].set(0.0)

    # Lambda gauge: (m,n)=(0,0) stays 0 everywhere.
    if idx00 is not None:
        gLcos = gLcos.at[:, idx00].set(0.0)
        gLsin = gLsin.at[:, idx00].set(0.0)

    return VMECState(
        layout=grad.layout,
        Rcos=gRcos,
        Rsin=gRsin,
        Zcos=gZcos,
        Zsin=gZsin,
        Lcos=gLcos,
        Lsin=gLsin,
    )


def _apply_preconditioner(
    grad: VMECState,
    static,
    *,
    kind: str,
    exponent: float = 1.0,
    radial_alpha: float = 0.0,
) -> VMECState:
    """Apply a simple diagonal preconditioner in (m,n) Fourier space.

    Parameters
    ----------
    kind:
        - ``"none"``: no preconditioning
        - ``"mode_diag"``: scale each (m,n) mode by ~(m^2 + (n*NFP)^2)^(-exponent)
        - ``"radial_tridi"``: apply a simple Dirichlet tri-diagonal smoother in s
        - ``"mode_diag+radial_tridi"``: apply both (order: mode, then radial)
    """
    kind = str(kind).strip().lower()
    if kind == "none":
        return grad

    kinds = [k.strip() for k in kind.replace("+", ",").split(",") if k.strip()]
    if not kinds:
        return grad

    exponent = float(exponent)
    if ("mode_diag" in kinds) and exponent <= 0.0:
        raise ValueError("preconditioner exponent must be > 0 for mode_diag")
    radial_alpha = float(radial_alpha)
    if ("radial_tridi" in kinds) and radial_alpha <= 0.0:
        raise ValueError("radial_alpha must be > 0 for radial_tridi")

    def _apply_mode_diag(g: VMECState) -> VMECState:
        m = jnp.asarray(static.modes.m)
        n = jnp.asarray(static.modes.n)
        nfp = float(static.cfg.nfp)
        k2 = m.astype(jnp.float64) ** 2 + (n.astype(jnp.float64) * nfp) ** 2
        # (1 + k2)^(-exponent) avoids singularity at (m,n)=(0,0).
        w = (1.0 + k2) ** (-exponent)
        w = w.astype(jnp.asarray(g.Rcos).dtype)

        def _scale(a):
            a = jnp.asarray(a)
            return a * w[None, :]

        return VMECState(
            layout=g.layout,
            Rcos=_scale(g.Rcos),
            Rsin=_scale(g.Rsin),
            Zcos=_scale(g.Zcos),
            Zsin=_scale(g.Zsin),
            Lcos=_scale(g.Lcos),
            Lsin=_scale(g.Lsin),
        )

    def _apply_radial_tridi(g: VMECState) -> VMECState:
        return VMECState(
            layout=g.layout,
            Rcos=_radial_tridi_smooth_dirichlet(g.Rcos, alpha=radial_alpha),
            Rsin=_radial_tridi_smooth_dirichlet(g.Rsin, alpha=radial_alpha),
            Zcos=_radial_tridi_smooth_dirichlet(g.Zcos, alpha=radial_alpha),
            Zsin=_radial_tridi_smooth_dirichlet(g.Zsin, alpha=radial_alpha),
            Lcos=_radial_tridi_smooth_dirichlet(g.Lcos, alpha=radial_alpha),
            Lsin=_radial_tridi_smooth_dirichlet(g.Lsin, alpha=radial_alpha),
        )

    g = grad
    for k in kinds:
        if k == "mode_diag":
            g = _apply_mode_diag(g)
        elif k == "radial_tridi":
            g = _apply_radial_tridi(g)
        else:
            raise ValueError(f"Unknown preconditioner kind={k!r}")
    return g


def solve_lambda_gd(
    state0: VMECState,
    static,
    *,
    phipf,
    chipf,
    signgs: int,
    lamscale,
    sqrtg: Any | None = None,
    max_iter: int = 50,
    step_size: float = 0.05,
    grad_tol: float | None = None,
    max_backtracks: int = 16,
    bt_factor: float = 0.5,
    jit_grad: bool = False,
    preconditioner: str = "none",
    precond_exponent: float = 1.0,
    precond_radial_alpha: float = 0.0,
    verbose: bool = True,
) -> SolveLambdaResult:
    """Solve for VMEC lambda (scaled coefficients) with fixed R/Z.

    Parameters
    ----------
    state0:
        Initial state. Only the lambda coefficients are updated.
    static:
        VMECStatic from :func:`vmec_jax.static.build_static`.
    phipf, chipf:
        1D flux functions (ns,) matching VMEC's `wout` meaning.
    signgs:
        Orientation (+1 or -1).
    lamscale:
        VMEC lambda scaling factor (see :func:`vmec_jax.field.lamscale_from_phips`).
    sqrtg:
        Optional signed Jacobian on the 3D grid. If provided (e.g. reconstructed from
        `wout` Nyquist coefficients), it is used for the objective and field formulas.
        Otherwise we use :func:`vmec_jax.geom.eval_geom`'s sqrtg.
    """
    if not has_jax():
        raise ImportError("solve_lambda_gd requires JAX (jax + jaxlib)")

    opts = validate_lambda_gd_options(
        max_iter=max_iter,
        max_backtracks=max_backtracks,
        bt_factor=bt_factor,
        preconditioner=preconditioner,
        precond_exponent=precond_exponent,
    )
    max_iter = opts.max_iter
    max_backtracks = opts.max_backtracks
    bt_factor = opts.bt_factor
    preconditioner = opts.preconditioner
    precond_exponent = opts.precond_exponent

    idx00 = _mode00_index(static.modes)

    # Metric depends only on R/Z, so compute it once.
    g0 = eval_geom(state0, static)
    gtt = jnp.asarray(g0.g_tt)
    gtp = jnp.asarray(g0.g_tp)
    gpp = jnp.asarray(g0.g_pp)

    sqrtg_use = jnp.asarray(g0.sqrtg if sqrtg is None else sqrtg)

    phipf = jnp.asarray(phipf)
    chipf = jnp.asarray(chipf)
    lamscale = jnp.asarray(lamscale)
    signgs = int(signgs)
    nfp = int(static.cfg.nfp)

    s = jnp.asarray(static.s)
    dtype_state = jnp.asarray(state0.Rcos).dtype
    zero_precond_diag = (
        jnp.zeros((int(s.shape[0]),), dtype=dtype_state),
        jnp.zeros((int(s.shape[0]),), dtype=dtype_state),
    )
    zero_tcon = jnp.zeros((int(s.shape[0]),), dtype=dtype_state)
    constraint_active_false = jnp.asarray(False)
    constraint_active_false = jnp.asarray(False)
    constraint_active_false = jnp.asarray(False)
    theta = jnp.asarray(static.grid.theta)
    zeta = jnp.asarray(static.grid.zeta)
    if s.shape[0] < 2:
        ds = jnp.asarray(1.0, dtype=s.dtype)
    else:
        ds = s[1] - s[0]
    dtheta_f, dzeta_f = angle_steps(ntheta=int(theta.shape[0]), nzeta=int(zeta.shape[0]))
    dtheta = jnp.asarray(dtheta_f, dtype=s.dtype)
    dzeta = jnp.asarray(dzeta_f, dtype=s.dtype)
    weight = ds * dtheta * dzeta

    def _wb_from_L(Lcos, Lsin):
        lam_u = eval_fourier_dtheta(Lcos, Lsin, static.basis, coeffs_internal=True)
        lam_v = eval_fourier_dzeta_phys(Lcos, Lsin, static.basis, coeffs_internal=True) / nfp
        bsupu, bsupv = bsup_from_sqrtg_lambda(
            sqrtg=sqrtg_use,
            lam_u=lam_u,
            lam_v=lam_v,
            phipf=phipf,
            chipf=chipf,
            signgs=signgs,
            lamscale=lamscale,
        )
        B2 = gtt * bsupu**2 + 2.0 * gtp * bsupu * bsupv + gpp * bsupv**2
        jac = signgs * sqrtg_use
        E_total = jnp.sum(0.5 * B2 * jac) * weight
        return E_total / (TWOPI * TWOPI)

    wb_and_grad = jax.value_and_grad(_wb_from_L, argnums=(0, 1))
    wb_only = _wb_from_L
    if jit_grad:
        wb_and_grad = jit(wb_and_grad)
        wb_only = jit(wb_only)

    Lcos = jnp.asarray(state0.Lcos)
    Lsin = jnp.asarray(state0.Lsin)
    Lcos, Lsin = _enforce_lambda_gauge(Lcos, Lsin, idx00=idx00)

    wb0, (gcos, gsin) = wb_and_grad(Lcos, Lsin)
    wb_history = [float(np.asarray(wb0))]
    grad_rms_history = []
    step_history = []
    grad_tol_eff: float | None = None

    for it in range(max_iter):
        # Optional mode-diagonal preconditioning for the lambda subproblem.
        if preconditioner == "mode_diag":
            m = jnp.asarray(static.modes.m)
            n = jnp.asarray(static.modes.n)
            k2 = m.astype(jnp.float64) ** 2 + (n.astype(jnp.float64) * float(static.cfg.nfp)) ** 2
            w = (1.0 + k2) ** (-precond_exponent)
            w = w.astype(jnp.asarray(Lcos).dtype)
            gcos_p = gcos * w[None, :]
            gsin_p = gsin * w[None, :]
        else:
            gcos_p = gcos
            gsin_p = gsin

        grad_rms = float(np.sqrt(np.mean(np.asarray(gcos_p) ** 2 + np.asarray(gsin_p) ** 2)))
        grad_rms_history.append(grad_rms)
        if grad_tol_eff is None:
            grad_tol_eff = _resolve_grad_tol(grad_tol, grad_rms0=grad_rms, dtype=np.asarray(Lcos).dtype)

        if verbose:
            print(f"[solve_lambda_gd] iter={it:03d} wb={wb_history[-1]:.8e} grad_rms={grad_rms:.3e}")

        if grad_rms < float(grad_tol_eff):
            break

        step = float(step_size)
        accepted = False

        for bt in range(max_backtracks + 1):
            if bt > 0:
                step *= bt_factor
            Lcos_t = Lcos - step * gcos_p
            Lsin_t = Lsin - step * gsin_p
            Lcos_t, Lsin_t = _enforce_lambda_gauge(Lcos_t, Lsin_t, idx00=idx00)
            wb_t = wb_only(Lcos_t, Lsin_t)
            if float(np.asarray(wb_t)) < wb_history[-1]:
                accepted = True
                Lcos, Lsin, wb0 = Lcos_t, Lsin_t, wb_t
                break

        step_history.append(step)

        if not accepted:
            if verbose:
                print("[solve_lambda_gd] line search failed to improve objective; stopping")
            break

        wb_history.append(float(np.asarray(wb0)))
        wb0, (gcos, gsin) = wb_and_grad(Lcos, Lsin)

    st = VMECState(
        layout=state0.layout,
        Rcos=state0.Rcos,
        Rsin=state0.Rsin,
        Zcos=state0.Zcos,
        Zsin=state0.Zsin,
        Lcos=Lcos,
        Lsin=Lsin,
    )
    diag: Dict[str, Any] = {
        "idx00": idx00,
        "grad_tol": None if grad_tol_eff is None else float(grad_tol_eff),
    }
    return SolveLambdaResult(
        state=st,
        n_iter=len(wb_history) - 1,
        wb_history=np.asarray(wb_history, dtype=float),
        grad_rms_history=np.asarray(grad_rms_history, dtype=float),
        step_history=np.asarray(step_history, dtype=float),
        diagnostics=diag,
    )


def solve_fixed_boundary_gd(
    state0: VMECState,
    static,
    *,
    phipf,
    chipf,
    signgs: int,
    lamscale,
    edge_Rcos: Any | None = None,
    edge_Rsin: Any | None = None,
    edge_Zcos: Any | None = None,
    edge_Zsin: Any | None = None,
    pressure: Any | None = None,
    gamma: float = 0.0,
    jacobian_penalty: float = 1e3,
    max_iter: int = 25,
    step_size: float = 5e-3,
    scale_rz: float = 1.0,
    scale_l: float = 1.0,
    grad_tol: float | None = None,
    max_backtracks: int = 16,
    bt_factor: float = 0.5,
    jit_grad: bool = False,
    preconditioner: str = "none",
    precond_exponent: float = 1.0,
    precond_radial_alpha: float = 0.0,
    differentiable: bool = False,
    stop_grad_in_update: bool = False,
    verbose: bool = True,
) -> SolveFixedBoundaryResult:
    """Minimize a VMEC-style energy objective over (R,Z,lambda) coefficients.

    This is the first "full" fixed-boundary solver step:
    - R/Z are evolved on interior surfaces only; the outer surface is held fixed.
    - Lambda gauge mode (0,0) is fixed to 0.

    The objective is::

        W = wb + wp/(gamma - 1)

    where ``wb`` is VMEC's normalized magnetic energy and
    ``wp = ∫ p dV /(2π)^2``.
    A soft penalty enforces a consistent Jacobian sign away from the axis.
    """
    if not has_jax():
        raise ImportError("solve_fixed_boundary_gd requires JAX (jax + jaxlib)")

    opts = validate_fixed_boundary_gd_options(
        max_iter=max_iter,
        max_backtracks=max_backtracks,
        bt_factor=bt_factor,
        gamma=gamma,
    )
    max_iter = opts.max_iter
    max_backtracks = opts.max_backtracks
    bt_factor = opts.bt_factor
    gamma = opts.gamma

    idx00 = _mode00_index(static.modes)

    phipf = jnp.asarray(phipf)
    chipf = jnp.asarray(chipf)
    lamscale = jnp.asarray(lamscale)
    signgs = int(signgs)
    nfp = int(static.cfg.nfp)

    s = jnp.asarray(static.s)
    theta = jnp.asarray(static.grid.theta)
    zeta = jnp.asarray(static.grid.zeta)
    if s.shape[0] < 2:
        ds = jnp.asarray(1.0, dtype=s.dtype)
    else:
        ds = s[1] - s[0]
    dtheta_f, dzeta_f = angle_steps(ntheta=int(theta.shape[0]), nzeta=int(zeta.shape[0]))
    dtheta = jnp.asarray(dtheta_f, dtype=s.dtype)
    dzeta = jnp.asarray(dzeta_f, dtype=s.dtype)
    weight = ds * dtheta * dzeta

    if pressure is None:
        pressure = jnp.zeros_like(s)
    pressure = jnp.asarray(pressure)
    validate_pressure_shape(tuple(pressure.shape), tuple(s.shape))

    edge_Rcos = jnp.asarray(edge_Rcos) if edge_Rcos is not None else jnp.asarray(state0.Rcos)[-1, :]
    edge_Rsin = jnp.asarray(edge_Rsin) if edge_Rsin is not None else jnp.asarray(state0.Rsin)[-1, :]
    edge_Zcos = jnp.asarray(edge_Zcos) if edge_Zcos is not None else jnp.asarray(state0.Zcos)[-1, :]
    edge_Zsin = jnp.asarray(edge_Zsin) if edge_Zsin is not None else jnp.asarray(state0.Zsin)[-1, :]

    def _wb_wp_from_geom(g) -> Tuple[Any, Any]:
        bsupu, bsupv = bsup_from_geom(g, phipf=phipf, chipf=chipf, nfp=nfp, signgs=signgs, lamscale=lamscale)
        B2 = b2_from_bsup(g, bsupu, bsupv)
        jac = signgs * g.sqrtg
        wb = (jnp.sum(0.5 * B2 * jac) * weight) / (TWOPI * TWOPI)
        wp = (jnp.sum(pressure[:, None, None] * jac) * weight) / (TWOPI * TWOPI)
        return wb, wp

    def _w_total_from_wb_wp(wb, wp) -> Any:
        return wb + wp / (gamma - 1.0)

    def _objective(state: VMECState) -> Any:
        # Softly enforce a consistent Jacobian sign away from the axis (s=0).
        g = eval_geom(state, static)
        wb, wp = _wb_wp_from_geom(g)
        w = _w_total_from_wb_wp(wb, wp)
        jac = signgs * g.sqrtg
        jac = jac.at[0, :, :].set(0.0)
        neg = jnp.minimum(jac, 0.0)
        penalty = float(jacobian_penalty) * jnp.mean(neg * neg)
        return w + penalty

    def _w_terms(state: VMECState) -> Tuple[Any, Any, Any]:
        g = eval_geom(state, static)
        wb, wp = _wb_wp_from_geom(g)
        return wb, wp, _w_total_from_wb_wp(wb, wp)

    obj_and_grad = jax.value_and_grad(_objective)
    w_terms = _w_terms
    if jit_grad:
        obj_and_grad = jit(obj_and_grad)
        w_terms = jit(w_terms)

    # Start from a constraint-satisfying state.
    state = _enforce_fixed_boundary_and_axis(
        state0,
        static,
        edge_Rcos=edge_Rcos,
        edge_Rsin=edge_Rsin,
        edge_Zcos=edge_Zcos,
        edge_Zsin=edge_Zsin,
        enforce_lambda_axis=False,
        idx00=idx00,
    )

    grad_tol_eff: float | None = None

    if differentiable:
        wb_history = []
        wp_history = []
        w_history = []
        grad_rms_history = []
        step_history = []

        def _grad_rms_jax(grad_state: VMECState):
            g = (
                jnp.asarray(grad_state.Rcos) ** 2
                + jnp.asarray(grad_state.Rsin) ** 2
                + jnp.asarray(grad_state.Zcos) ** 2
                + jnp.asarray(grad_state.Zsin) ** 2
                + jnp.asarray(grad_state.Lcos) ** 2
                + jnp.asarray(grad_state.Lsin) ** 2
            )
            return jnp.sqrt(jnp.mean(g))

        for _ in range(max_iter):
            wb_t, wp_t, w_t = w_terms(state)
            w_history.append(w_t)
            wb_history.append(wb_t)
            wp_history.append(wp_t)

            obj_t, grad_t = obj_and_grad(state)
            grad_t = _mask_grad_for_constraints(grad_t, static, idx00=idx00)
            grad_t = _apply_preconditioner(
                grad_t,
                static,
                kind=preconditioner,
                exponent=precond_exponent,
                radial_alpha=precond_radial_alpha,
            )
            if stop_grad_in_update:
                grad_t = jax.lax.stop_gradient(grad_t)
            grad_rms_history.append(_grad_rms_jax(grad_t))
            step_history.append(jnp.asarray(step_size, dtype=jnp.asarray(state.Rcos).dtype))

            state = _update_state_gd(state, grad_t, step=step_size, scale_rz=scale_rz, scale_l=scale_l)
            state = _enforce_fixed_boundary_and_axis(
                state,
                static,
                edge_Rcos=edge_Rcos,
                edge_Rsin=edge_Rsin,
                edge_Zcos=edge_Zcos,
                edge_Zsin=edge_Zsin,
                idx00=idx00,
            )
    else:
        wb0, wp0, w0 = w_terms(state)
        wb0 = float(np.asarray(wb0))
        wp0 = float(np.asarray(wp0))
        w0 = float(np.asarray(w0))
        wb_history = [wb0]
        wp_history = [wp0]
        grad_rms_history = []
        step_history = []

        obj0, grad0 = obj_and_grad(state)
        obj0 = float(np.asarray(obj0))
        w_history = [obj0]

        for it in range(max_iter):
            grad0m = _mask_grad_for_constraints(grad0, static, idx00=idx00)
            grad_raw = grad0m
            grad0m = _apply_preconditioner(
                grad0m,
                static,
                kind=preconditioner,
                exponent=precond_exponent,
                radial_alpha=precond_radial_alpha,
            )
            grad_rms = _grad_rms_state(grad0m)
            grad_rms_history.append(grad_rms)
            if grad_tol_eff is None:
                grad_tol_eff = _resolve_grad_tol(grad_tol, grad_rms0=grad_rms, dtype=np.asarray(state.Rcos).dtype)

            if verbose:
                print(f"[solve_fixed_boundary_gd] iter={it:03d} w={w_history[-1]:.8e} grad_rms={grad_rms:.3e}")

            if grad_rms < float(grad_tol_eff):
                break

            step = float(step_size)
            accepted = False

            def _try_line_search(grad_step):
                step_local = float(step_size)
                for bt in range(max_backtracks + 1):
                    if bt > 0:
                        step_local *= bt_factor
                    trial = _update_state_gd(state, grad_step, step=step_local, scale_rz=scale_rz, scale_l=scale_l)
                    trial = _enforce_fixed_boundary_and_axis(
                        trial,
                        static,
                        edge_Rcos=edge_Rcos,
                        edge_Rsin=edge_Rsin,
                        edge_Zcos=edge_Zcos,
                        edge_Zsin=edge_Zsin,
                        idx00=idx00,
                    )
                    obj_t = _objective(trial)
                    obj_t = float(np.asarray(obj_t))
                    if np.isfinite(obj_t) and obj_t < w_history[-1]:
                        return True, trial, obj_t, step_local
                return False, None, None, step_local

            accepted, trial, obj_t, step = _try_line_search(grad0m)
            if not accepted and preconditioner != "none":
                accepted, trial, obj_t, step = _try_line_search(grad_raw)
                if accepted and verbose:
                    print("[solve_fixed_boundary_gd] fallback to unpreconditioned gradient")

            step_history.append(step)

            if not accepted:
                if verbose:
                    print("[solve_fixed_boundary_gd] line search failed to improve objective; stopping")
                break

            state = trial
            obj0 = obj_t

            wb_t, wp_t, _w_t = w_terms(state)
            w_history.append(obj0)
            wb_history.append(float(np.asarray(wb_t)))
            wp_history.append(float(np.asarray(wp_t)))

            obj0, grad0 = obj_and_grad(state)

    diag: Dict[str, Any] = {
        "idx00": idx00,
        "signgs": signgs,
        "gamma": gamma,
        "jacobian_penalty": float(jacobian_penalty),
        "scale_rz": float(scale_rz),
        "scale_l": float(scale_l),
        "preconditioner": str(preconditioner),
        "precond_exponent": float(precond_exponent),
        "precond_radial_alpha": float(precond_radial_alpha),
        "grad_tol": None if grad_tol_eff is None else float(grad_tol_eff),
    }
    if differentiable:
        return SolveFixedBoundaryResult(
            state=state,
            n_iter=len(w_history),
            w_history=jnp.asarray(w_history),
            wb_history=jnp.asarray(wb_history),
            wp_history=jnp.asarray(wp_history),
            grad_rms_history=jnp.asarray(grad_rms_history),
            step_history=jnp.asarray(step_history),
            diagnostics=diag,
        )
    return SolveFixedBoundaryResult(
        state=state,
        n_iter=len(w_history) - 1,
        w_history=np.asarray(w_history, dtype=float),
        wb_history=np.asarray(wb_history, dtype=float),
        wp_history=np.asarray(wp_history, dtype=float),
        grad_rms_history=np.asarray(grad_rms_history, dtype=float),
        step_history=np.asarray(step_history, dtype=float),
        diagnostics=diag,
    )


def solve_fixed_boundary_lbfgs(
    state0: VMECState,
    static,
    *,
    phipf,
    chipf,
    signgs: int,
    lamscale,
    edge_Rcos: Any | None = None,
    edge_Rsin: Any | None = None,
    edge_Zcos: Any | None = None,
    edge_Zsin: Any | None = None,
    pressure: Any | None = None,
    gamma: float = 0.0,
    history_size: int = 10,
    max_iter: int = 40,
    step_size: float = 1.0,
    grad_tol: float | None = None,
    max_backtracks: int = 12,
    bt_factor: float = 0.5,
    jit_grad: bool = False,
    preconditioner: str = "none",
    precond_exponent: float = 1.0,
    precond_radial_alpha: float = 0.0,
    verbose: bool = True,
) -> SolveFixedBoundaryResult:
    """Fixed-boundary solve using L-BFGS (no external deps).

    This solver minimizes::

        W = wb + wp/(gamma - 1)

    with:

    - fixed R/Z edge coefficients (prescribed boundary),
    - simple axis regularity,
    - lambda gauge (0,0)=0.
    """
    if not has_jax():
        raise ImportError("solve_fixed_boundary_lbfgs requires JAX (jax + jaxlib)")

    opts = validate_fixed_boundary_lbfgs_options(
        history_size=history_size,
        max_iter=max_iter,
        max_backtracks=max_backtracks,
        bt_factor=bt_factor,
        gamma=gamma,
    )
    history_size = opts.history_size
    max_iter = opts.max_iter
    max_backtracks = opts.max_backtracks
    bt_factor = opts.bt_factor
    gamma = opts.gamma

    idx00 = _mode00_index(static.modes)

    phipf = jnp.asarray(phipf)
    chipf = jnp.asarray(chipf)
    lamscale = jnp.asarray(lamscale)
    signgs = int(signgs)
    nfp = int(static.cfg.nfp)

    s = jnp.asarray(static.s)
    theta = jnp.asarray(static.grid.theta)
    zeta = jnp.asarray(static.grid.zeta)
    if s.shape[0] < 2:
        ds = jnp.asarray(1.0, dtype=s.dtype)
    else:
        ds = s[1] - s[0]
    dtheta_f, dzeta_f = angle_steps(ntheta=int(theta.shape[0]), nzeta=int(zeta.shape[0]))
    dtheta = jnp.asarray(dtheta_f, dtype=s.dtype)
    dzeta = jnp.asarray(dzeta_f, dtype=s.dtype)
    weight = ds * dtheta * dzeta

    if pressure is None:
        pressure = jnp.zeros_like(s)
    pressure = jnp.asarray(pressure)
    validate_pressure_shape(tuple(pressure.shape), tuple(s.shape))

    edge_Rcos = jnp.asarray(edge_Rcos) if edge_Rcos is not None else jnp.asarray(state0.Rcos)[-1, :]
    edge_Rsin = jnp.asarray(edge_Rsin) if edge_Rsin is not None else jnp.asarray(state0.Rsin)[-1, :]
    edge_Zcos = jnp.asarray(edge_Zcos) if edge_Zcos is not None else jnp.asarray(state0.Zcos)[-1, :]
    edge_Zsin = jnp.asarray(edge_Zsin) if edge_Zsin is not None else jnp.asarray(state0.Zsin)[-1, :]

    def _wb_wp_from_geom(g) -> Tuple[Any, Any]:
        bsupu, bsupv = bsup_from_geom(g, phipf=phipf, chipf=chipf, nfp=nfp, signgs=signgs, lamscale=lamscale)
        B2 = b2_from_bsup(g, bsupu, bsupv)
        jac = signgs * g.sqrtg
        wb = (jnp.sum(0.5 * B2 * jac) * weight) / (TWOPI * TWOPI)
        wp = (jnp.sum(pressure[:, None, None] * jac) * weight) / (TWOPI * TWOPI)
        return wb, wp

    def _w_total_from_wb_wp(wb, wp) -> Any:
        return wb + wp / (gamma - 1.0)

    def _w_only(state: VMECState) -> Any:
        g = eval_geom(state, static)
        wb, wp = _wb_wp_from_geom(g)
        return _w_total_from_wb_wp(wb, wp)

    def _w_terms_and_jacmin(state: VMECState) -> Tuple[Any, Any, Any, Any]:
        g = eval_geom(state, static)
        wb, wp = _wb_wp_from_geom(g)
        w = _w_total_from_wb_wp(wb, wp)
        jac = signgs * g.sqrtg
        if jac.shape[0] <= 1:
            jac_min = jnp.min(jac)
        else:
            jac_min = jnp.min(jac[1:, :, :])
        return wb, wp, w, jac_min

    w_and_grad = jax.value_and_grad(_w_only)
    w_terms = _w_terms_and_jacmin
    if jit_grad:
        w_and_grad = jit(w_and_grad)
        w_terms = jit(w_terms)

    # Start from a constraint-satisfying state.
    state = _enforce_fixed_boundary_and_axis(
        state0,
        static,
        edge_Rcos=edge_Rcos,
        edge_Rsin=edge_Rsin,
        edge_Zcos=edge_Zcos,
        edge_Zsin=edge_Zsin,
        enforce_lambda_axis=False,
        idx00=idx00,
    )

    wb0, wp0, w0, jacmin0 = w_terms(state)
    w0 = float(np.asarray(w0))
    wb0 = float(np.asarray(wb0))
    wp0 = float(np.asarray(wp0))
    jacmin0 = float(np.asarray(jacmin0))
    if not np.isfinite(w0) or jacmin0 <= 0.0:
        raise ValueError("Initial state has invalid Jacobian sign or non-finite energy")

    w_history = [w0]
    wb_history = [wb0]
    wp_history = [wp0]
    grad_rms_history = []
    step_history = []

    w_val, grad = w_and_grad(state)
    grad = _mask_grad_for_constraints(grad, static, idx00=idx00, mask_lambda_axis=False)
    grad = _apply_preconditioner(
        grad,
        static,
        kind=preconditioner,
        exponent=precond_exponent,
        radial_alpha=precond_radial_alpha,
    )

    x = pack_state(state)
    g_flat = pack_state(grad)

    s_hist: list[Any] = []
    y_hist: list[Any] = []

    step0 = float(step_size)
    grad_tol_eff: float | None = None

    for it in range(max_iter):
        grad_rms = _grad_rms_state(grad)
        grad_rms_history.append(grad_rms)
        if grad_tol_eff is None:
            grad_tol_eff = _resolve_grad_tol(grad_tol, grad_rms0=grad_rms, dtype=np.asarray(state.Rcos).dtype)

        if verbose:
            print(f"[solve_fixed_boundary_lbfgs] iter={it:03d} w={w_history[-1]:.8e} grad_rms={grad_rms:.3e}")

        if grad_rms < float(grad_tol_eff):
            break

        p_flat = _lbfgs_two_loop_direction(g_flat, s_hist, y_hist)
        p_flat, _gtp, _fallback_to_descent = _ensure_descent_direction(g_flat, p_flat)

        accepted = False
        step = step0

        x_old = x
        g_old = g_flat

        for bt in range(max_backtracks + 1):
            if bt > 0:
                step *= bt_factor
            x_try = x_old + jnp.asarray(step, dtype=x_old.dtype) * p_flat
            st_try = unpack_state(x_try, state.layout)
            st_try = _enforce_fixed_boundary_and_axis(
                st_try,
                static,
                edge_Rcos=edge_Rcos,
                edge_Rsin=edge_Rsin,
                edge_Zcos=edge_Zcos,
                edge_Zsin=edge_Zsin,
                enforce_lambda_axis=False,
                idx00=idx00,
            )

            wb_t, wp_t, w_t, jacmin_t = w_terms(st_try)
            w_tf = float(np.asarray(w_t))
            jacmin_tf = float(np.asarray(jacmin_t))
            if np.isfinite(w_tf) and jacmin_tf > 0.0 and w_tf < w_history[-1]:
                state = st_try
                x = pack_state(state)
                accepted = True
                break

        step_history.append(step)

        if not accepted:
            if verbose:
                print("[solve_fixed_boundary_lbfgs] line search failed; stopping")
            break

        # New value/grad at accepted state.
        wb_t, wp_t, w_t, _jacmin_t = w_terms(state)
        w_history.append(float(np.asarray(w_t)))
        wb_history.append(float(np.asarray(wb_t)))
        wp_history.append(float(np.asarray(wp_t)))

        w_val, grad_new = w_and_grad(state)
        grad_new = _mask_grad_for_constraints(grad_new, static, idx00=idx00)
        grad_new = _apply_preconditioner(
            grad_new,
            static,
            kind=preconditioner,
            exponent=precond_exponent,
            radial_alpha=precond_radial_alpha,
        )
        g_flat_new = pack_state(grad_new)

        s_k = x - x_old
        y_k = g_flat_new - g_old
        ys = float(np.asarray(jnp.dot(y_k, s_k)))
        if np.isfinite(ys) and ys > _resolve_lbfgs_curvature_tol(s_k, y_k):
            s_hist.append(s_k)
            y_hist.append(y_k)
            if len(s_hist) > history_size:
                s_hist.pop(0)
                y_hist.pop(0)

        grad = grad_new
        g_flat = g_flat_new
        step0 = float(step)

    diag: Dict[str, Any] = {
        "idx00": idx00,
        "signgs": signgs,
        "gamma": gamma,
        "history_size": int(history_size),
        "preconditioner": str(preconditioner),
        "precond_exponent": float(precond_exponent),
        "precond_radial_alpha": float(precond_radial_alpha),
        "grad_tol": None if grad_tol_eff is None else float(grad_tol_eff),
    }
    return SolveFixedBoundaryResult(
        state=state,
        n_iter=len(w_history) - 1,
        w_history=np.asarray(w_history, dtype=float),
        wb_history=np.asarray(wb_history, dtype=float),
        wp_history=np.asarray(wp_history, dtype=float),
        grad_rms_history=np.asarray(grad_rms_history, dtype=float),
        step_history=np.asarray(step_history, dtype=float),
        diagnostics=diag,
    )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _WoutLikeVmecForces:
    """Minimal `wout`-like container for VMEC force/residual kernels."""

    nfp: int
    mpol: int
    ntor: int
    lasym: bool
    signgs: int

    phipf: Any  # (ns,)
    phips: Any  # (ns,)
    chipf: Any  # (ns,)  (VMEC `wout` half-mesh averaged convention)
    pres: Any  # (ns,)  (half mesh, VMEC internal units mu0*Pa)
    mass: Any | None = None  # (ns,) mass profile on half mesh (VMEC internal units)
    gamma: float | None = None
    ncurr: int = 0
    lcurrent: bool = True
    icurv: Any | None = None  # (ns,) integrated toroidal current profile
    flux_is_internal: bool = True
    phipf_internal: Any | None = None
    chipf_internal: Any | None = None
    chips_eff: Any | None = None

    def tree_flatten(self):
        children = (
            self.phipf,
            self.phips,
            self.chipf,
            self.pres,
            self.mass,
            self.icurv,
            self.phipf_internal,
            self.chipf_internal,
            self.chips_eff,
        )
        aux = (
            int(self.nfp),
            int(self.mpol),
            int(self.ntor),
            bool(self.lasym),
            int(self.signgs),
            None if self.gamma is None else float(self.gamma),
            int(self.ncurr),
            bool(self.lcurrent),
            bool(self.flux_is_internal),
        )
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (
            nfp,
            mpol,
            ntor,
            lasym,
            signgs,
            gamma,
            ncurr,
            lcurrent,
            flux_is_internal,
        ) = aux_data
        return cls(
            nfp=int(nfp),
            mpol=int(mpol),
            ntor=int(ntor),
            lasym=bool(lasym),
            signgs=int(signgs),
            gamma=gamma,
            ncurr=int(ncurr),
            lcurrent=bool(lcurrent),
            flux_is_internal=bool(flux_is_internal),
            phipf=children[0],
            phips=children[1],
            chipf=children[2],
            pres=children[3],
            mass=children[4],
            icurv=children[5],
            phipf_internal=children[6],
            chipf_internal=children[7],
            chips_eff=children[8],
        )


def solve_fixed_boundary_lbfgs_vmec_residual(
    state0: VMECState,
    static,
    *,
    indata,
    signgs: int,
    w_rz: float = 1.0,
    w_l: float = 1.0,
    include_constraint_force: bool = True,
    objective_scale: float | None = None,
    apply_m1_constraints: bool = True,
    history_size: int = 10,
    max_iter: int = 40,
    step_size: float = 1.0,
    scale_rz: float = 1.0,
    scale_l: float = 1.0,
    grad_tol: float | None = None,
    max_backtracks: int = 12,
    bt_factor: float = 0.5,
    jit_grad: bool = False,
    preconditioner: str = "none",
    precond_exponent: float = 1.0,
    precond_radial_alpha: float = 0.0,
    verbose: bool = True,
) -> SolveVmecResidualResult:
    """Fixed-boundary solve by minimizing a VMEC-style force-residual objective.

    The objective follows the parity pipeline
    ``bcovar -> forces -> tomnsps -> sum-of-squares of Fourier residual blocks``,
    using VMEC's ``getfsq`` conventions (post-``tomnsps`` ``scalxc`` scaling,
    optional converged-iteration m=1 constraints, and R/Z edge exclusion).

    For parity, build ``static`` with ``vmec_angle_grid(...)`` (see
    ``vmec_jax.vmec_tomnsp``). This solver does not include VMEC's
    iteration-dependent switching logic (e.g. ``lforbal`` triggering); it
    provides a differentiable objective suitable for regression and initial
    end-to-end parity.

    """
    if not has_jax():
        raise ImportError("solve_fixed_boundary_lbfgs_vmec_residual requires JAX (jax + jaxlib)")

    opts = validate_residual_lbfgs_options(
        w_rz=w_rz,
        w_l=w_l,
        objective_scale=objective_scale,
        scale_rz=scale_rz,
        scale_l=scale_l,
        history_size=history_size,
        max_iter=max_iter,
        max_backtracks=max_backtracks,
        bt_factor=bt_factor,
    )
    w_rz = opts.w_rz
    w_l = opts.w_l
    objective_scale = opts.objective_scale
    scale_rz = opts.scale_rz
    scale_l = opts.scale_l
    history_size = opts.history_size
    max_iter = opts.max_iter
    max_backtracks = opts.max_backtracks
    bt_factor = opts.bt_factor

    idx00 = _mode00_index(static.modes)
    signgs = int(signgs)

    from .energy import flux_profiles_from_indata
    from .vmec_forces import vmec_forces_rz_from_wout, vmec_residual_internal_from_kernels
    from .vmec_residue import (
        vmec_force_norms_from_bcovar_dynamic,
    )
    from .vmec_tomnsp import vmec_trig_tables

    s = jnp.asarray(static.s)

    flux = flux_profiles_from_indata(indata, s, signgs=signgs)
    chipf_wout = jnp.asarray(flux.chipf)

    phips = jnp.asarray(flux.phips)
    if phips.shape[0] >= 1:
        phips = phips.at[0].set(0.0)

    # VMEC mass profile uses the boundary r00 coefficient and phips scaling.
    from .boundary import boundary_from_indata

    boundary = boundary_from_indata(indata, static.modes)
    r00 = float(np.asarray(boundary.R_cos)[int(idx00)]) if int(idx00) >= 0 else float(np.asarray(boundary.R_cos)[0])
    gamma = float(indata.get_float("GAMMA", 0.0))
    lrfp = bool(indata.get_bool("LRFP", False))
    chips = _half_mesh_from_full_mesh(chipf_wout) if lrfp else None
    mass = _mass_half_mesh_from_indata(
        indata=indata,
        s_full=s,
        phips=phips,
        r00=r00,
        gamma=gamma,
        lrfp=lrfp,
        chips=chips,
    )

    pres = _pressure_half_mesh_from_indata(indata=indata, s_full=s)
    ncurr = int(indata.get_int("NCURR", 0))
    icurv = _icurv_full_mesh_from_indata(indata=indata, s_full=s, signgs=signgs)
    phipf_internal, chipf_internal, chips_eff = _vmec_force_flux_profiles(
        phipf=jnp.asarray(flux.phipf),
        chipf=chipf_wout,
        signgs=signgs,
        flux_is_internal=True,
    )

    wout_like = _WoutLikeVmecForces(
        nfp=int(static.cfg.nfp),
        mpol=int(static.cfg.mpol),
        ntor=int(static.cfg.ntor),
        lasym=bool(static.cfg.lasym),
        signgs=signgs,
        phipf=jnp.asarray(flux.phipf),
        phips=phips,
        chipf=chipf_wout,
        pres=pres,
        mass=mass,
        gamma=gamma,
        ncurr=ncurr,
        lcurrent=True,
        icurv=icurv,
        phipf_internal=phipf_internal,
        chipf_internal=chipf_internal,
        chips_eff=chips_eff,
    )
    trig = getattr(static, "trig_vmec", None)
    if trig is None:
        trig = vmec_trig_tables(
            ntheta=int(static.cfg.ntheta),
            nzeta=int(static.cfg.nzeta),
            nfp=int(wout_like.nfp),
            mmax=int(wout_like.mpol) - 1,
            nmax=int(wout_like.ntor),
            lasym=bool(wout_like.lasym),
            dtype=jnp.asarray(state0.Rcos).dtype,
        )
    objective_scale_f = float(objective_scale) if objective_scale is not None else None
    ftol_target = max(0.0, float(indata.get_float("FTOL", 0.0)))

    constraint_tcon0: float | None = None
    if bool(include_constraint_force):
        # VMEC2000 default is `TCON0=1` (readin.f).
        constraint_tcon0 = float(indata.get_float("TCON0", 1.0))
    apply_lforbal = bool(indata.get_bool("LFORBAL", False)) if indata is not None else False
    apply_lforbal = bool(indata.get_bool("LFORBAL", False)) if indata is not None else False
    mask_pack = getattr(static, "tomnsps_masks", None)

    def _fsq2_terms_and_jacmin(state: VMECState, zero_m1_zforce: Any):
        k = vmec_forces_rz_from_wout(
            state=state,
            static=static,
            wout=wout_like,
            indata=None,
            constraint_tcon0=constraint_tcon0,
            use_vmec_synthesis=True,
            trig=trig,
        )
        rzl = vmec_residual_internal_from_kernels(
            k,
            cfg_ntheta=int(static.cfg.ntheta),
            cfg_nzeta=int(static.cfg.nzeta),
            wout=wout_like,
            trig=trig,
            apply_lforbal=apply_lforbal,
            include_edge=False,
            masks=mask_pack,
        )
        norms = vmec_force_norms_from_bcovar_dynamic(bc=k.bc, trig=trig, s=s, signgs=signgs)
        terms = _assemble_residual_objective_terms(
            frzl=rzl,
            norms=norms,
            s=s,
            w_rz=w_rz,
            w_l=w_l,
            zero_m1_zforce=zero_m1_zforce,
            lconm1=bool(getattr(static.cfg, "lconm1", True)),
            apply_m1_constraints=bool(apply_m1_constraints),
            zero_m1_after_m1_constraints=False,
            include_edge=False,
            apply_scalxc=True,
            zero_edge_rz_blocks=False,
            objective_scale=objective_scale_f,
        )

        jac = signgs * jnp.asarray(k.bc.jac.sqrtg)
        jac_min = jnp.min(jac) if jac.shape[0] <= 1 else jnp.min(jac[1:, :, :])
        return terms.fsqr2, terms.fsqz2, terms.fsql2, terms.w, jac_min

    def _w_only(state: VMECState, zero_m1_zforce: Any):
        return _fsq2_terms_and_jacmin(state, zero_m1_zforce)[3]

    w_and_grad = jax.value_and_grad(_w_only)
    w_terms = _fsq2_terms_and_jacmin
    if jit_grad:
        w_and_grad = jit(w_and_grad)
        w_terms = jit(w_terms)

    edge_Rcos = jnp.asarray(state0.Rcos)[-1, :]
    edge_Rsin = jnp.asarray(state0.Rsin)[-1, :]
    edge_Zcos = jnp.asarray(state0.Zcos)[-1, :]
    edge_Zsin = jnp.asarray(state0.Zsin)[-1, :]

    state = _enforce_fixed_boundary_and_axis(
        state0,
        static,
        edge_Rcos=edge_Rcos,
        edge_Rsin=edge_Rsin,
        edge_Zcos=edge_Zcos,
        edge_Zsin=edge_Zsin,
        idx00=idx00,
    )

    zero_m1 = jnp.asarray(1.0, dtype=jnp.asarray(state0.Rcos).dtype)
    fsqr2_0, fsqz2_0, fsql2_0, w0, jacmin0 = w_terms(state, zero_m1)
    w0 = float(np.asarray(w0))
    jacmin0 = float(np.asarray(jacmin0))
    if not np.isfinite(w0):
        raise ValueError("Initial state has non-finite residual objective")
    if jacmin0 <= 0.0 and verbose:
        print("[solve_fixed_boundary_lbfgs_vmec_residual] warning: initial Jacobian has non-positive entries")

    if objective_scale_f is None:
        # Auto-scale the objective to be O(1) on the initial iterate.
        objective_scale_f = 1.0 / max(abs(w0), 1.0)

        # Rebuild the objective closures with the now-fixed scale.
        def _fsq2_terms_and_jacmin(state: VMECState, zero_m1_zforce: Any):  # type: ignore[no-redef]
            k = vmec_forces_rz_from_wout(
                state=state,
                static=static,
                wout=wout_like,
                indata=None,
                constraint_tcon0=constraint_tcon0,
                use_vmec_synthesis=True,
                trig=trig,
            )
            rzl = vmec_residual_internal_from_kernels(
                k,
                cfg_ntheta=int(static.cfg.ntheta),
                cfg_nzeta=int(static.cfg.nzeta),
                wout=wout_like,
                trig=trig,
                apply_lforbal=apply_lforbal,
                include_edge=False,
                masks=mask_pack,
            )
            norms = vmec_force_norms_from_bcovar_dynamic(bc=k.bc, trig=trig, s=s, signgs=signgs)
            terms = _assemble_residual_objective_terms(
                frzl=rzl,
                norms=norms,
                s=s,
                w_rz=w_rz,
                w_l=w_l,
                zero_m1_zforce=zero_m1_zforce,
                lconm1=bool(getattr(static.cfg, "lconm1", True)),
                apply_m1_constraints=bool(apply_m1_constraints),
                zero_m1_after_m1_constraints=False,
                include_edge=False,
                apply_scalxc=True,
                zero_edge_rz_blocks=False,
                objective_scale=objective_scale_f,
            )

            jac = signgs * jnp.asarray(k.bc.jac.sqrtg)
            jac_min = jnp.min(jac) if jac.shape[0] <= 1 else jnp.min(jac[1:, :, :])
            return terms.fsqr2, terms.fsqz2, terms.fsql2, terms.w, jac_min

        def _w_only(state: VMECState, zero_m1_zforce: Any):  # type: ignore[no-redef]
            return _fsq2_terms_and_jacmin(state, zero_m1_zforce)[3]

        w_and_grad = jax.value_and_grad(_w_only)
        w_terms = _fsq2_terms_and_jacmin
        if jit_grad:
            w_and_grad = jit(w_and_grad)
            w_terms = jit(w_terms)

        fsqr2_0, fsqz2_0, fsql2_0, w0, jacmin0 = w_terms(state, zero_m1)
        w0 = float(np.asarray(w0))

    w_history = [w0]
    fsqr2_history = [float(np.asarray(fsqr2_0))]
    fsqz2_history = [float(np.asarray(fsqz2_0))]
    fsql2_history = [float(np.asarray(fsql2_0))]
    grad_rms_history = []
    step_history = []

    w_val, grad = w_and_grad(state, zero_m1)
    grad = _mask_grad_for_constraints(grad, static, idx00=idx00, mask_lambda_axis=False)
    grad = _apply_preconditioner(
        grad,
        static,
        kind=preconditioner,
        exponent=precond_exponent,
        radial_alpha=precond_radial_alpha,
    )
    sr = jnp.asarray(scale_rz, dtype=jnp.asarray(grad.Rcos).dtype)
    sl = jnp.asarray(scale_l, dtype=jnp.asarray(grad.Lcos).dtype)
    grad = VMECState(
        layout=grad.layout,
        Rcos=jnp.asarray(grad.Rcos) * sr,
        Rsin=jnp.asarray(grad.Rsin) * sr,
        Zcos=jnp.asarray(grad.Zcos) * sr,
        Zsin=jnp.asarray(grad.Zsin) * sr,
        Lcos=jnp.asarray(grad.Lcos) * sl,
        Lsin=jnp.asarray(grad.Lsin) * sl,
    )

    x = pack_state(state)
    g_flat = pack_state(grad)

    s_hist: list[Any] = []
    y_hist: list[Any] = []

    step0 = float(step_size)
    grad_tol_eff: float | None = None
    zero_m1_fsqz_target = float(ftol_target)

    for it in range(max_iter):
        grad_rms = _grad_rms_state(grad)
        grad_rms_history.append(grad_rms)
        if grad_tol_eff is None:
            grad_tol_eff = _resolve_grad_tol(grad_tol, grad_rms0=grad_rms, dtype=np.asarray(state.Rcos).dtype)

        if verbose:
            print(
                f"[solve_fixed_boundary_lbfgs_vmec_residual] iter={it:03d} w={w_history[-1]:.8e} grad_rms={grad_rms:.3e}"
            )

        if grad_rms < float(grad_tol_eff):
            break

        p_flat = _lbfgs_two_loop_direction(g_flat, s_hist, y_hist)
        p_flat, _gtp, _fallback_to_descent = _ensure_descent_direction(g_flat, p_flat)

        accepted = False
        step = step0
        best_w = np.inf
        best_state = None
        best_step = None
        best_fsqr2 = None
        best_fsqz2 = None
        best_fsql2 = None

        x_old = x
        g_old = g_flat

        zero_m1 = jnp.asarray(
            1.0 if ((len(step_history) == 0) or (fsqz2_history[-1] < zero_m1_fsqz_target)) else 0.0,
            dtype=jnp.asarray(state.Rcos).dtype,
        )
        for bt in range(max_backtracks + 1):
            if bt > 0:
                step *= bt_factor
            x_try = x_old + jnp.asarray(step, dtype=x_old.dtype) * p_flat
            st_try = unpack_state(x_try, state.layout)
            st_try = _enforce_fixed_boundary_and_axis(
                st_try,
                static,
                edge_Rcos=edge_Rcos,
                edge_Rsin=edge_Rsin,
                edge_Zcos=edge_Zcos,
                edge_Zsin=edge_Zsin,
                idx00=idx00,
            )

            fsqr2_t, fsqz2_t, fsql2_t, w_t, jacmin_t = w_terms(st_try, zero_m1)
            w_tf = float(np.asarray(w_t))
            jacmin_tf = float(np.asarray(jacmin_t))
            if np.isfinite(w_tf) and w_tf < best_w:
                best_w = w_tf
                best_state = st_try
                best_step = step
                best_fsqr2 = float(np.asarray(fsqr2_t))
                best_fsqz2 = float(np.asarray(fsqz2_t))
                best_fsql2 = float(np.asarray(fsql2_t))
            if np.isfinite(w_tf) and jacmin_tf > 0.0 and w_tf < w_history[-1]:
                state = st_try
                x = pack_state(state)
                accepted = True
                fsqr2_accept = float(np.asarray(fsqr2_t))
                fsqz2_accept = float(np.asarray(fsqz2_t))
                fsql2_accept = float(np.asarray(fsql2_t))
                break

        step_history.append(step)

        if not accepted:
            if best_state is not None and np.isfinite(best_w):
                if verbose:
                    print("[solve_fixed_boundary_lbfgs_vmec_residual] line search failed; accepting best finite step")
                state = best_state
                x = pack_state(state)
                w_t = best_w
                fsqr2_accept = best_fsqr2 if best_fsqr2 is not None else float(np.asarray(fsqr2_t))
                fsqz2_accept = best_fsqz2 if best_fsqz2 is not None else float(np.asarray(fsqz2_t))
                fsql2_accept = best_fsql2 if best_fsql2 is not None else float(np.asarray(fsql2_t))
                step_history[-1] = best_step
            else:
                if verbose:
                    print("[solve_fixed_boundary_lbfgs_vmec_residual] line search failed; stopping")
                break

        w_history.append(float(np.asarray(w_t)))
        fsqr2_history.append(fsqr2_accept)
        fsqz2_history.append(fsqz2_accept)
        fsql2_history.append(fsql2_accept)

        w_val, grad_new = w_and_grad(state, zero_m1)
        grad_new = _mask_grad_for_constraints(grad_new, static, idx00=idx00, mask_lambda_axis=False)
        grad_new = _apply_preconditioner(
            grad_new,
            static,
            kind=preconditioner,
            exponent=precond_exponent,
            radial_alpha=precond_radial_alpha,
        )
        g_flat_new = pack_state(grad_new)

        s_k = x - x_old
        y_k = g_flat_new - g_old
        ys = float(np.asarray(jnp.dot(y_k, s_k)))
        if np.isfinite(ys) and ys > _resolve_lbfgs_curvature_tol(s_k, y_k):
            s_hist.append(s_k)
            y_hist.append(y_k)
            if len(s_hist) > history_size:
                s_hist.pop(0)
                y_hist.pop(0)

        grad = grad_new
        g_flat = g_flat_new
        step0 = float(step)

    diag: Dict[str, Any] = {
        "idx00": idx00,
        "signgs": signgs,
        "w_rz": float(w_rz),
        "w_l": float(w_l),
        "objective_scale": float(objective_scale_f),
        "include_constraint_force": bool(include_constraint_force),
        "scale_rz": float(scale_rz),
        "scale_l": float(scale_l),
        "apply_m1_constraints": bool(apply_m1_constraints),
        "history_size": int(history_size),
        "preconditioner": str(preconditioner),
        "precond_exponent": float(precond_exponent),
        "precond_radial_alpha": float(precond_radial_alpha),
        "grad_tol": None if grad_tol_eff is None else float(grad_tol_eff),
        "zero_m1_fsqz_thresh": float(zero_m1_fsqz_target),
    }
    res_final = SolveVmecResidualResult(
        state=state,
        n_iter=len(w_history) - 1,
        w_history=np.asarray(w_history, dtype=float),
        fsqr2_history=np.asarray(fsqr2_history, dtype=float),
        fsqz2_history=np.asarray(fsqz2_history, dtype=float),
        fsql2_history=np.asarray(fsql2_history, dtype=float),
        grad_rms_history=np.asarray(grad_rms_history, dtype=float),
        step_history=np.asarray(step_history, dtype=float),
        diagnostics=diag,
    )
    return res_final


def solve_fixed_boundary_gn_vmec_residual(
    state0: VMECState,
    static,
    *,
    indata,
    signgs: int,
    w_rz: float = 1.0,
    w_l: float = 1.0,
    include_constraint_force: bool = True,
    apply_m1_constraints: bool = True,
    objective_scale: float | None = None,
    damping: float | None = None,
    damping_increase: float = 10.0,
    damping_decrease: float = 0.5,
    max_damping: float | None = None,
    max_retries: int = 6,
    zero_m1_iters: int | None = None,
    zero_m1_fsqz_thresh: float | None = None,
    max_iter: int = 20,
    cg_tol: float | None = None,
    cg_maxiter: int = 80,
    step_size: float = 1.0,
    max_backtracks: int = 12,
    bt_factor: float = 0.5,
    jit_kernels: bool = True,
    verbose: bool = True,
) -> SolveVmecResidualResult:
    """Fixed-boundary solve using a Gauss-Newton (normal-equations) step on VMEC residuals.

    This treats the VMEC residual blocks returned by `tomnsps` as a least-squares
    problem and solves (approximately) for a step `dx` using conjugate gradients:

        (Jᵀ J + damping * I) dx = -Jᵀ r

    where `r(state)` is the stacked residual vector and `J` is its Jacobian.

    The residual vector uses the same conventions as `vmec_jax.vmec_residue`
    (post-`tomnsps` `scalxc` scaling, optional m=1 constraints, and R/Z edge
    exclusion) so the objective is consistent with the scalar residual definitions.
    """
    if not has_jax():
        raise ImportError("solve_fixed_boundary_gn_vmec_residual requires JAX (jax + jaxlib)")
    opts = validate_residual_gn_options(
        damping=damping,
        damping_increase=damping_increase,
        damping_decrease=damping_decrease,
        max_damping=max_damping,
        max_retries=max_retries,
        zero_m1_iters=zero_m1_iters,
        zero_m1_fsqz_thresh=zero_m1_fsqz_thresh,
        w_rz=w_rz,
        w_l=w_l,
        max_iter=max_iter,
        cg_maxiter=cg_maxiter,
        max_backtracks=max_backtracks,
        bt_factor=bt_factor,
        objective_scale=objective_scale,
    )
    damping = opts.damping
    damping_increase = opts.damping_increase
    damping_decrease = opts.damping_decrease
    max_damping_eff = opts.max_damping_eff
    max_retries = opts.max_retries
    zero_m1_iters_eff = opts.zero_m1_iters_eff
    zero_m1_fsqz_thresh = opts.zero_m1_fsqz_thresh
    w_rz = opts.w_rz
    w_l = opts.w_l
    max_iter = opts.max_iter
    cg_maxiter = opts.cg_maxiter
    max_backtracks = opts.max_backtracks
    bt_factor = opts.bt_factor
    objective_scale = opts.objective_scale

    constraint_tcon0: float | None = None
    if bool(include_constraint_force):
        constraint_tcon0 = float(indata.get_float("TCON0", 1.0))
    apply_lforbal = bool(indata.get_bool("LFORBAL", False)) if indata is not None else False
    apply_lforbal = bool(indata.get_bool("LFORBAL", False)) if indata is not None else False
    apply_lforbal = bool(indata.get_bool("LFORBAL", False)) if indata is not None else False
    apply_lforbal = bool(indata.get_bool("LFORBAL", False)) if indata is not None else False

    signgs = int(signgs)
    idx00 = _mode00_index(static.modes)
    ftol_target = max(0.0, float(indata.get_float("FTOL", 0.0)))
    zero_m1_fsqz_thresh_eff = float(ftol_target) if zero_m1_fsqz_thresh is None else float(zero_m1_fsqz_thresh)

    from .energy import flux_profiles_from_indata
    from .vmec_forces import vmec_forces_rz_from_wout, vmec_residual_internal_from_kernels
    from .vmec_residue import (
        vmec_force_norms_from_bcovar_dynamic,
    )
    from .vmec_tomnsp import vmec_trig_tables

    try:
        from jax.scipy.sparse.linalg import cg  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("solve_fixed_boundary_gn_vmec_residual requires jax.scipy.sparse.linalg.cg") from e

    s = jnp.asarray(static.s)
    flux = flux_profiles_from_indata(indata, s, signgs=signgs)
    chipf_wout = jnp.asarray(flux.chipf)

    phips = jnp.asarray(flux.phips)
    if phips.shape[0] >= 1:
        phips = phips.at[0].set(0.0)

    from .boundary import boundary_from_indata

    boundary = boundary_from_indata(indata, static.modes)
    r00 = float(np.asarray(boundary.R_cos)[int(idx00)]) if int(idx00) >= 0 else float(np.asarray(boundary.R_cos)[0])
    gamma = float(indata.get_float("GAMMA", 0.0))
    lrfp = bool(indata.get_bool("LRFP", False))
    chips = _half_mesh_from_full_mesh(chipf_wout) if lrfp else None
    mass = _mass_half_mesh_from_indata(
        indata=indata,
        s_full=s,
        phips=phips,
        r00=r00,
        gamma=gamma,
        lrfp=lrfp,
        chips=chips,
    )

    pres = _pressure_half_mesh_from_indata(indata=indata, s_full=s)
    ncurr = int(indata.get_int("NCURR", 0))
    icurv = _icurv_full_mesh_from_indata(indata=indata, s_full=s, signgs=signgs)
    phipf_internal, chipf_internal, chips_eff = _vmec_force_flux_profiles(
        phipf=jnp.asarray(flux.phipf),
        chipf=chipf_wout,
        signgs=signgs,
        flux_is_internal=True,
    )

    wout_like = _WoutLikeVmecForces(
        nfp=int(static.cfg.nfp),
        mpol=int(static.cfg.mpol),
        ntor=int(static.cfg.ntor),
        lasym=bool(static.cfg.lasym),
        signgs=signgs,
        phipf=jnp.asarray(flux.phipf),
        phips=phips,
        chipf=chipf_wout,
        pres=pres,
        mass=mass,
        gamma=gamma,
        ncurr=ncurr,
        lcurrent=True,
        icurv=icurv,
        phipf_internal=phipf_internal,
        chipf_internal=chipf_internal,
        chips_eff=chips_eff,
    )
    trig = getattr(static, "trig_vmec", None)
    if trig is None:
        trig = vmec_trig_tables(
            ntheta=int(static.cfg.ntheta),
            nzeta=int(static.cfg.nzeta),
            nfp=int(wout_like.nfp),
            mmax=int(wout_like.mpol) - 1,
            nmax=int(wout_like.ntor),
            lasym=bool(wout_like.lasym),
            dtype=jnp.asarray(state0.Rcos).dtype,
        )

    edge_Rcos = jnp.asarray(state0.Rcos)[-1, :]
    edge_Rsin = jnp.asarray(state0.Rsin)[-1, :]
    edge_Zcos = jnp.asarray(state0.Zcos)[-1, :]
    edge_Zsin = jnp.asarray(state0.Zsin)[-1, :]

    def _project_step(d: VMECState) -> VMECState:
        return _mask_grad_for_constraints(d, static, idx00=idx00, mask_lambda_axis=True)

    def _enforce_state(st: VMECState) -> VMECState:
        return _enforce_fixed_boundary_and_axis(
            st,
            static,
            edge_Rcos=edge_Rcos,
            edge_Rsin=edge_Rsin,
            edge_Zcos=edge_Zcos,
            edge_Zsin=edge_Zsin,
            enforce_lambda_axis=True,
            idx00=idx00,
        )

    mask_pack = getattr(static, "tomnsps_masks", None)

    def _residual_blocks(state: VMECState, zero_m1_zforce: Any):
        k = vmec_forces_rz_from_wout(
            state=state,
            static=static,
            wout=wout_like,
            indata=None,
            constraint_tcon0=constraint_tcon0,
            use_vmec_synthesis=True,
            trig=trig,
        )
        rzl = vmec_residual_internal_from_kernels(
            k,
            cfg_ntheta=int(static.cfg.ntheta),
            cfg_nzeta=int(static.cfg.nzeta),
            wout=wout_like,
            trig=trig,
            apply_lforbal=apply_lforbal,
            include_edge=False,
            masks=mask_pack,
        )
        norms = vmec_force_norms_from_bcovar_dynamic(bc=k.bc, trig=trig, s=s, signgs=signgs)
        terms = _assemble_residual_objective_terms(
            frzl=rzl,
            norms=norms,
            s=s,
            w_rz=w_rz,
            w_l=w_l,
            zero_m1_zforce=zero_m1_zforce,
            lconm1=bool(getattr(static.cfg, "lconm1", True)),
            apply_m1_constraints=bool(apply_m1_constraints),
            zero_m1_after_m1_constraints=True,
            include_edge=True,
            apply_scalxc=True,
            zero_edge_rz_blocks=True,
            objective_scale=None,
        )
        return terms.frzl, terms.fsqr2, terms.fsqz2, terms.fsql2, terms.norms

    def _residual_vec(state: VMECState, zero_m1_zforce: Any) -> Any:
        frzl, *_vals = _residual_blocks(state, zero_m1_zforce)
        norms = _vals[-1]
        return _residual_objective_vector(frzl=frzl, norms=norms, w_rz=w_rz, w_l=w_l)

    def _obj_terms(state: VMECState, zero_m1_zforce: Any):
        _frzl, fsqr2, fsqz2, fsql2, _norms = _residual_blocks(state, zero_m1_zforce)
        w = (w_rz * (fsqr2 + fsqz2)) + (w_l * fsql2)
        return fsqr2, fsqz2, fsql2, w

    if bool(jit_kernels):
        _residual_vec_jit = jit(_residual_vec)
        _obj_terms_jit = jit(_obj_terms)
    else:
        _residual_vec_jit = _residual_vec
        _obj_terms_jit = _obj_terms

    state = _enforce_state(state0)
    zero_m1 = jnp.asarray(1.0, dtype=jnp.asarray(state0.Rcos).dtype)
    fsqr2_0, fsqz2_0, fsql2_0, w0 = _obj_terms_jit(state, zero_m1)
    w0_f = float(np.asarray(w0))
    if not np.isfinite(w0_f):
        raise ValueError("Initial state has non-finite residual objective")

    scale_f = float(objective_scale) if objective_scale is not None else (1.0 / max(abs(w0_f), 1.0))

    w_history = [float(scale_f * w0_f)]
    fsqr2_history = [float(np.asarray(fsqr2_0))]
    fsqz2_history = [float(np.asarray(fsqz2_0))]
    fsql2_history = [float(np.asarray(fsql2_0))]
    grad_rms_history = []
    step_history = []

    for it in range(int(max_iter)):
        zero_m1_active = (len(step_history) < int(zero_m1_iters_eff)) or (len(step_history) == 0)
        zero_m1_active = zero_m1_active or (fsqz2_history[-1] < float(zero_m1_fsqz_thresh_eff))
        zero_m1 = jnp.asarray(
            1.0 if zero_m1_active else 0.0,
            dtype=jnp.asarray(state.Rcos).dtype,
        )
        r, pullback = jax.vjp(_residual_vec_jit, state, zero_m1)
        # Gradient of 0.5*||r||^2 is J^T r.
        g_state = pullback(r)[0]
        g_state = _project_step(g_state)
        grad_rms_history.append(_grad_rms_state(g_state))

        b_flat = -pack_state(g_state)
        dtype_state = np.asarray(state.Rcos).dtype
        current_w = float(w_history[-1])
        cg_tol_it = _resolve_cg_tol(
            cg_tol,
            current_obj=current_w,
            initial_obj=float(w_history[0]),
            target_obj=float(ftol_target),
            dtype=dtype_state,
        )
        g_norm_sq = float(np.asarray(jnp.dot(b_flat, b_flat)))
        if np.isfinite(g_norm_sq) and g_norm_sq > _dtype_tiny(dtype_state):
            zero_tangent = jnp.zeros_like(zero_m1)
            jg = jax.jvp(_residual_vec_jit, (state, zero_m1), (g_state, zero_tangent))[1]
            jt_jg = pullback(jg)[0]
            jt_jg = _project_step(jt_jg)
            curvature_num = float(np.asarray(jnp.dot(pack_state(g_state), pack_state(jt_jg))))
            curvature_scale = max(0.0, curvature_num / max(g_norm_sq, _dtype_tiny(dtype_state)))
        else:
            curvature_scale = 0.0
        damping_it = _resolve_lm_damping(damping, curvature_scale=curvature_scale, dtype=dtype_state)

        accepted = False
        step = float(step_size)
        w_curr = w_history[-1]
        retry = 0
        while True:
            dmp = float(damping_it)

            def _matvec(v_flat):
                v_state = unpack_state(v_flat, state.layout)
                v_state = _project_step(v_state)
                zero_tangent = jnp.zeros_like(zero_m1)
                jv = jax.jvp(_residual_vec_jit, (state, zero_m1), (v_state, zero_tangent))[1]
                jt_jv = pullback(jv)[0]
                jt_jv = _project_step(jt_jv)
                if dmp != 0.0:
                    jt_jv = VMECState(
                        layout=jt_jv.layout,
                        Rcos=jt_jv.Rcos + dmp * v_state.Rcos,
                        Rsin=jt_jv.Rsin + dmp * v_state.Rsin,
                        Zcos=jt_jv.Zcos + dmp * v_state.Zcos,
                        Zsin=jt_jv.Zsin + dmp * v_state.Zsin,
                        Lcos=jt_jv.Lcos + dmp * v_state.Lcos,
                        Lsin=jt_jv.Lsin + dmp * v_state.Lsin,
                    )
                return pack_state(jt_jv)

            dx_flat, _info = cg(_matvec, b_flat, tol=float(cg_tol_it), maxiter=int(cg_maxiter))
            dx_state = unpack_state(dx_flat, state.layout)
            dx_state = _project_step(dx_state)

            step = float(step_size)
            for bt in range(int(max_backtracks) + 1):
                if bt > 0:
                    step *= float(bt_factor)
                st_try = VMECState(
                    layout=state.layout,
                    Rcos=jnp.asarray(state.Rcos)
                    + jnp.asarray(step, dtype=jnp.asarray(state.Rcos).dtype) * jnp.asarray(dx_state.Rcos),
                    Rsin=jnp.asarray(state.Rsin)
                    + jnp.asarray(step, dtype=jnp.asarray(state.Rsin).dtype) * jnp.asarray(dx_state.Rsin),
                    Zcos=jnp.asarray(state.Zcos)
                    + jnp.asarray(step, dtype=jnp.asarray(state.Zcos).dtype) * jnp.asarray(dx_state.Zcos),
                    Zsin=jnp.asarray(state.Zsin)
                    + jnp.asarray(step, dtype=jnp.asarray(state.Zsin).dtype) * jnp.asarray(dx_state.Zsin),
                    Lcos=jnp.asarray(state.Lcos)
                    + jnp.asarray(step, dtype=jnp.asarray(state.Lcos).dtype) * jnp.asarray(dx_state.Lcos),
                    Lsin=jnp.asarray(state.Lsin)
                    + jnp.asarray(step, dtype=jnp.asarray(state.Lsin).dtype) * jnp.asarray(dx_state.Lsin),
                )
                st_try = _enforce_state(st_try)
                fsqr2_t, fsqz2_t, fsql2_t, w_t = _obj_terms_jit(st_try, zero_m1)
                w_tf = float(np.asarray(w_t))
                w_scaled = float(scale_f * w_tf)
                if np.isfinite(w_scaled) and w_scaled < w_curr:
                    state = st_try
                    accepted = True
                    w_history.append(w_scaled)
                    fsqr2_history.append(float(np.asarray(fsqr2_t)))
                    fsqz2_history.append(float(np.asarray(fsqz2_t)))
                    fsql2_history.append(float(np.asarray(fsql2_t)))
                    break

            if accepted:
                # Levenberg-Marquardt style: relax damping after success.
                damping_it = max(damping_it * damping_decrease, 0.0)
                break

            if retry >= max_retries or damping_it >= max_damping_eff:
                break
            # Increase damping and try again from the same state.
            damping_it = min(max_damping_eff, damping_it * damping_increase)
            retry += 1

        if not accepted:
            # Robust fallback: take a small steepest-descent step on 0.5*||r||^2
            # using the already-computed gradient g_state = J^T r.
            dx_state = unpack_state(b_flat, state.layout)  # b_flat = -grad_flat
            dx_state = _project_step(dx_state)
            step = float(step_size)
            for bt in range(int(max_backtracks) + 1):
                if bt > 0:
                    step *= float(bt_factor)
                st_try = VMECState(
                    layout=state.layout,
                    Rcos=jnp.asarray(state.Rcos)
                    + jnp.asarray(step, dtype=jnp.asarray(state.Rcos).dtype) * jnp.asarray(dx_state.Rcos),
                    Rsin=jnp.asarray(state.Rsin)
                    + jnp.asarray(step, dtype=jnp.asarray(state.Rsin).dtype) * jnp.asarray(dx_state.Rsin),
                    Zcos=jnp.asarray(state.Zcos)
                    + jnp.asarray(step, dtype=jnp.asarray(state.Zcos).dtype) * jnp.asarray(dx_state.Zcos),
                    Zsin=jnp.asarray(state.Zsin)
                    + jnp.asarray(step, dtype=jnp.asarray(state.Zsin).dtype) * jnp.asarray(dx_state.Zsin),
                    Lcos=jnp.asarray(state.Lcos)
                    + jnp.asarray(step, dtype=jnp.asarray(state.Lcos).dtype) * jnp.asarray(dx_state.Lcos),
                    Lsin=jnp.asarray(state.Lsin)
                    + jnp.asarray(step, dtype=jnp.asarray(state.Lsin).dtype) * jnp.asarray(dx_state.Lsin),
                )
                st_try = _enforce_state(st_try)
                fsqr2_t, fsqz2_t, fsql2_t, w_t = _obj_terms_jit(st_try, zero_m1)
                w_tf = float(np.asarray(w_t))
                w_scaled = float(scale_f * w_tf)
                if np.isfinite(w_scaled) and w_scaled < w_curr:
                    state = st_try
                    accepted = True
                    w_history.append(w_scaled)
                    fsqr2_history.append(float(np.asarray(fsqr2_t)))
                    fsqz2_history.append(float(np.asarray(fsqz2_t)))
                    fsql2_history.append(float(np.asarray(fsql2_t)))
                    break

        step_history.append(step)
        if verbose:
            print(
                f"[solve_fixed_boundary_gn_vmec_residual] iter={it:03d} w={w_history[-1]:.8e} "
                f"step={step:.3e} accepted={accepted} damping={damping_it:.3e} cg_tol={cg_tol_it:.3e} retries={retry}"
            )

        if not accepted:
            break

    diag = {
        "idx00": idx00,
        "signgs": signgs,
        "w_rz": float(w_rz),
        "w_l": float(w_l),
        "objective_scale": float(scale_f),
        "apply_m1_constraints": bool(apply_m1_constraints),
        "damping": None if damping is None else float(damping),
        "damping_mode": "adaptive" if damping is None else "fixed",
        "cg_tol": None if cg_tol is None else float(cg_tol),
        "cg_tol_mode": "adaptive" if cg_tol is None else "fixed",
        "cg_maxiter": int(cg_maxiter),
        "zero_m1_iters": None if zero_m1_iters is None else int(zero_m1_iters),
        "zero_m1_fsqz_thresh": float(zero_m1_fsqz_thresh_eff),
    }
    return SolveVmecResidualResult(
        state=state,
        n_iter=len(w_history) - 1,
        w_history=np.asarray(w_history, dtype=float),
        fsqr2_history=np.asarray(fsqr2_history, dtype=float),
        fsqz2_history=np.asarray(fsqz2_history, dtype=float),
        fsql2_history=np.asarray(fsql2_history, dtype=float),
        grad_rms_history=np.asarray(grad_rms_history, dtype=float),
        step_history=np.asarray(step_history, dtype=float),
        diagnostics=diag,
    )


def solve_fixed_boundary_residual_iter(
    state0: VMECState,
    static,
    *,
    indata,
    signgs: int,
    ftol: float | None = None,
    max_iter: int = 50,
    step_size: float = 1.0,
    initial_flip_sign: float = 1.0,
    include_constraint_force: bool = True,
    apply_m1_constraints: bool = True,
    precond_radial_alpha: float = 0.5,
    precond_lambda_alpha: float = 0.5,
    mode_diag_exponent: float = 0.0,
    auto_flip_force: bool = True,
    divide_by_scalxc_for_update: bool = False,
    lambda_update_scale: float = 1.0,
    enforce_vmec_lambda_axis: bool = False,
    vmec2000_control: bool = False,
    strict_update: bool = True,
    backtracking: bool = False,
    limit_dt_from_force: bool = False,
    limit_update_rms: bool = False,
    reference_mode: bool = False,
    use_restart_triggers: bool | None = None,
    vmecpp_restart: bool = False,
    stage_prev_fsq: float | None = None,
    stage_transition_factor: float = 50.0,
    stage_transition_scale: float = 0.5,
    use_direct_fallback: bool | None = None,
    verbose: bool = True,
    verbose_vmec2000_table: bool = True,
    jit_forces: bool = True,
    jit_warmup_iters: int = 0,
    jit_precompile: bool = False,
    use_scan: bool = False,
    precompile_only: bool = False,
    resume_state: dict | None = None,
    scan_minimal_default: bool | None = None,
    light_history: bool | None = None,
    resume_state_mode: str | None = None,
    fsq_total_target: float | None = None,
    host_update_assembly: bool | None = None,
    preconditioner_use_precomputed_tridi: bool | None = None,
    preconditioner_use_lax_tridi: bool | None = None,
    adjoint_trace: bool = False,
    adjoint_trace_mode: str = "full",
    external_field_provider_kind: str | None = None,
    external_field_provider_static: Any = None,
    external_field_provider_params: Any = None,
    free_boundary_activate_fsq: float | None = None,
    state_only: bool = False,
    return_final_force_payload: bool = False,
) -> SolveVmecResidualResult:
    """VMEC-style fixed-point update loop using preconditioned force residuals."""
    _solve_wall_start = time.perf_counter()
    if not has_jax():
        raise ImportError("solve_fixed_boundary_residual_iter requires JAX (jax + jaxlib)")

    timing_env = os.getenv("VMEC_JAX_TIMING", "").strip().lower()
    timing_enabled = timing_env not in ("", "0", "false", "no")
    timing_detail_env = os.getenv("VMEC_JAX_TIMING_DETAIL", "").strip().lower()
    timing_detail_enabled = timing_enabled and timing_detail_env not in ("", "0", "false", "no")
    _setup_phase_timings = {
        "setup_static_grid_rebuild": 0.0,
        "setup_freeb_policy": 0.0,
        "setup_boundary_profiles": 0.0,
        "setup_cache_key_hash": 0.0,
        "setup_ptau_constants": 0.0,
        "setup_index_constants": 0.0,
        "setup_update_constants": 0.0,
    }

    def _setup_timer_start() -> float | None:
        return time.perf_counter() if bool(timing_enabled) else None

    def _record_setup_timing(key: str, start: float | None) -> None:
        if start is not None:
            _setup_phase_timings[key] = float(_setup_phase_timings.get(key, 0.0)) + (
                time.perf_counter() - float(start)
            )

    def _device_get_floats(*vals):
        """Batch host materialization for scalar diagnostics.

        In the VMEC2000 parity (non-scan) loop we still need host scalars to
        drive Python control flow (TimeStepControl / restarts / printing).
        Pulling them one-by-one forces repeated synchronization; batch them.
        """

        return tuple(float(v) for v in jax.device_get(vals))

    opts = validate_residual_iteration_options(
        max_iter=max_iter,
        step_size=step_size,
        precompile_only=precompile_only,
        signgs=signgs,
        lambda_update_scale=lambda_update_scale,
        enforce_vmec_lambda_axis=enforce_vmec_lambda_axis,
        vmec2000_control=vmec2000_control,
        reference_mode=reference_mode,
        limit_dt_from_force=limit_dt_from_force,
        limit_update_rms=limit_update_rms,
        backtracking=backtracking,
        strict_update=strict_update,
        jit_precompile=jit_precompile,
        use_scan=use_scan,
    )
    max_iter = opts.max_iter
    step_size = opts.step_size
    precompile_only = opts.precompile_only
    host_update_assembly = _host_update_assembly_policy(
        requested=host_update_assembly,
        use_scan=opts.use_scan,
        backend_name=jax.default_backend(),
        state_has_tracer=_tree_has_tracer(state0),
        allow_accelerator=os.getenv("VMEC_JAX_HOST_UPDATE_ON_ACCELERATOR", "").strip().lower()
        in ("1", "true", "yes", "on"),
    ).enabled
    adjoint_trace = bool(adjoint_trace)
    adjoint_trace_mode = _normalize_adjoint_trace_mode(adjoint_trace_mode)
    (
        preconditioner_use_precomputed_tridi_policy,
        preconditioner_use_lax_tridi_policy,
    ) = _resolve_preconditioner_tridi_policies(
        use_precomputed=preconditioner_use_precomputed_tridi,
    )

    def _adjoint_trace_array(value):
        return _materialize_adjoint_trace_array(value, mode=adjoint_trace_mode)

    signgs = opts.signgs
    fsq_total_target = None if fsq_total_target is None else max(0.0, float(fsq_total_target))
    lambda_update_scale = opts.lambda_update_scale
    enforce_vmec_lambda_axis = opts.enforce_vmec_lambda_axis
    vmec2000_control = opts.vmec2000_control
    badjac_config = _parse_bad_jacobian_config(os.environ)
    badjac_mode = badjac_config.mode
    badjac_use_state = badjac_config.use_state
    dump_ptau_state = badjac_config.dump_ptau_state
    light_history = _resolve_light_history(light_history, env_value=os.getenv("VMEC_JAX_LIGHT_HISTORY", "0"))
    resume_state_mode = _normalize_resume_state_mode(resume_state_mode)
    badjac_state_probe = badjac_config.state_probe
    badjac_initial_state_probe_iters = int(badjac_config.initial_state_probe_iters)
    ptau_tol = badjac_config.ptau_tol
    ptau_tol_rel = badjac_config.ptau_tol_rel
    reference_mode = opts.reference_mode
    jit_precompile = opts.jit_precompile
    restart_flags = _resolve_restart_flags(
        use_restart_triggers=use_restart_triggers,
        use_direct_fallback=use_direct_fallback,
        vmecpp_restart=vmecpp_restart,
    )
    use_restart_triggers = restart_flags.use_restart_triggers
    use_direct_fallback = restart_flags.use_direct_fallback
    vmecpp_restart = restart_flags.vmecpp_restart
    verbose_vmec2000_table = bool(verbose_vmec2000_table)
    # Allow automatic fallback to the non-scan path when scan diverges.
    # On GPU/TPU, the non-scan fallback uses a Python loop with per-iteration
    # device→host synchronization, which is catastrophically slow (~74 ms/iter
    # vs ~4 ms/iter in the scan path). Disable scan fallback by default on
    # non-CPU backends, unless the user explicitly sets VMEC_JAX_SCAN_FALLBACK=1.
    scan_fallback_policy = _scan_fallback_policy(
        backend_name=_scan_backend_name(),
        enabled_env=os.getenv("VMEC_JAX_SCAN_FALLBACK"),
        iters_env=os.getenv("VMEC_JAX_SCAN_FALLBACK_ITERS", "50"),
        badjac_limit_env=os.getenv("VMEC_JAX_SCAN_FALLBACK_BJAC_LIMIT", "10"),
        fsq_abs_env=os.getenv("VMEC_JAX_SCAN_FALLBACK_FSQ_ABS", "1.0e-2"),
        accept_frac_env=os.getenv("VMEC_JAX_SCAN_FALLBACK_ACCEPTED_FRAC", "0.5"),
        fsq_factor_env=os.getenv("VMEC_JAX_SCAN_FALLBACK_FSQ_FACTOR", "50"),
        improve_env=os.getenv("VMEC_JAX_SCAN_FALLBACK_IMPROVE", "0.1"),
    )
    scan_fallback_enabled = scan_fallback_policy.enabled
    scan_fallback_iters = scan_fallback_policy.iters
    scan_fallback_badjac_limit = scan_fallback_policy.badjac_limit
    scan_fallback_fsq_abs = scan_fallback_policy.fsq_abs
    scan_fallback_accept_frac = scan_fallback_policy.accept_frac
    scan_fallback_fsq_factor = scan_fallback_policy.fsq_factor
    scan_fallback_improve = scan_fallback_policy.improve
    stage_transition_factor = float(stage_transition_factor)
    stage_transition_scale = float(stage_transition_scale)
    if stage_transition_factor <= 0.0 or stage_transition_scale <= 0.0:
        stage_prev_fsq = None

    def _converged_residuals_host(*, fsqr: float, fsqz: float, fsql: float) -> bool:
        return _residual_convergence_flags(
            fsqr=fsqr,
            fsqz=fsqz,
            fsql=fsql,
            ftol=ftol,
            fsq_total_target=fsq_total_target,
        )[2]

    if use_scan and vmec2000_control and auto_flip_force:
        auto_flip_force = False
    jit_forces = bool(jit_forces)
    use_scan = bool(use_scan)
    # Default to chunked scan to reduce per-iteration host sync overhead.
    # Respect explicit non-scan requests (e.g., parity comparators).
    chunked_scan_config = _resolve_chunked_scan_config(
        use_scan=bool(use_scan),
        state_has_tracer=_tree_has_tracer(state0),
        scan_fallback_enabled=bool(scan_fallback_enabled),
        chunked_env=os.getenv("VMEC_JAX_VMEC2000_CHUNKED", "1"),
    )
    force_chunked_scan = chunked_scan_config.force_chunked_scan
    scan_fallback_enabled = chunked_scan_config.scan_fallback_enabled
    differentiating_scan = chunked_scan_config.differentiating_scan
    limit_dt_from_force = opts.limit_dt_from_force
    limit_update_rms = opts.limit_update_rms
    backtracking = opts.backtracking
    strict_update = opts.strict_update
    dump_history_config = _resolve_dump_history_config(
        env=os.environ,
        jit_forces=bool(jit_forces),
        light_history=bool(light_history),
        heavy_dump_envs=_HEAVY_DUMP_ENVS,
        light_dump_envs=_LIGHT_DUMP_ENVS,
    )
    dumps_enabled = dump_history_config.dumps_enabled
    dump_any = dump_history_config.dump_any
    if dump_history_config.disabled_jit_for_dumps:
        if verbose:
            print("[solve_fixed_boundary_residual_iter] jit_forces disabled (debug dumps enabled)")
    jit_forces = dump_history_config.jit_forces
    light_history = dump_history_config.light_history
    track_history = dump_history_config.track_history

    def _pack_resume_state(base: dict[str, Any], heavy: dict[str, Any] | None = None):
        return _pack_resume_state_record(base=base, heavy=heavy, mode=resume_state_mode)

    from .energy import flux_profiles_from_indata
    from .static import build_static
    from .boundary import boundary_from_indata
    from .init_guess import (
        _recompute_axis_from_boundary,
        _recompute_axis_from_state_vmec,
        _read_axis_coeffs,
        initial_guess_from_boundary,
    )
    from .vmec_forces import vmec_forces_rz_from_wout, vmec_residual_internal_from_kernels
    from .vmec_residue import (
        vmec_apply_m1_constraints,
        vmec_apply_scalxc_to_tomnsps,
        vmec_force_norms_from_bcovar_dynamic,
        vmec_gcx2_from_tomnsps,
        vmec_gcx2_from_tomnsps_np,
        vmec_scalxc_from_s,
        vmec_wint_from_trig,
        vmec_zero_m1_zforce,
    )
    from .vmec_jacobian import vmec_half_mesh_jacobian_from_state
    from .vmec_tomnsp import TomnspsRZL, vmec_angle_grid, vmec_trig_tables
    from .free_boundary import NestorRuntimeState, nestor_external_only_step

    # VMEC2000 evaluates the force kernels on VMEC's internal
    # angle grid. In particular, when `lasym=False`, VMEC uses a reduced theta
    # grid (stellarator symmetry) for the force pipeline. Rebuild `static`
    # using `vmec_angle_grid(...)` so the force terms do not mix full-grid and
    # VMEC-grid arrays (which triggers broadcasting errors and parity drift).
    _t_setup_static_grid = _setup_timer_start()
    cfg = static.cfg
    grid_vmec = vmec_angle_grid(
        ntheta=int(cfg.ntheta),
        nzeta=int(cfg.nzeta),
        nfp=int(cfg.nfp),
        lasym=bool(cfg.lasym),
    )
    reuse_static = False
    try:
        theta_curr = np.asarray(static.grid.theta)
        zeta_curr = np.asarray(static.grid.zeta)
        reuse_static = (
            int(static.grid.nfp) == int(grid_vmec.nfp)
            and theta_curr.shape == np.asarray(grid_vmec.theta).shape
            and zeta_curr.shape == np.asarray(grid_vmec.zeta).shape
            and np.allclose(theta_curr, np.asarray(grid_vmec.theta))
            and np.allclose(zeta_curr, np.asarray(grid_vmec.zeta))
        )
    except Exception:
        reuse_static = False
    if not reuse_static:
        static = build_static(
            cfg,
            grid=grid_vmec,
            mgrid_metadata=getattr(static, "mgrid_metadata", None),
            free_boundary_extcur=getattr(static, "free_boundary_extcur", None),
        )
    _record_setup_timing("setup_static_grid_rebuild", _t_setup_static_grid)
    # Free-boundary control + coupling path:
    # VMEC-style ivac/ivacskip cadence with edge bsqvac coupling.
    _t_setup_freeb_policy = _setup_timer_start()
    free_boundary_enabled = bool(getattr(cfg, "lfreeb", False))
    free_boundary_provider_kind = (
        ""
        if external_field_provider_kind is None
        else str(external_field_provider_kind).strip().lower()
    )
    direct_free_boundary_provider = free_boundary_provider_kind in ("direct_coils", "coils", "coil")
    freeb_nvacskip = max(1, int(getattr(cfg, "nvacskip", int(getattr(cfg, "nfp", 1)))))
    freeb_nvskip0 = max(1, freeb_nvacskip)
    freeb_couple_env = os.getenv("VMEC_JAX_FREEB_COUPLE_EDGE", "1").strip().lower()
    freeb_couple_edge = bool(free_boundary_enabled) and (freeb_couple_env not in ("", "0", "false", "no"))
    if free_boundary_enabled and use_scan:
        # WP2 free-boundary coupling is currently wired through the VMEC2000
        # control (non-scan) path, including ivacskip-driven reuse.
        use_scan = False
    freeb_sample_env = os.getenv("VMEC_JAX_FREEB_SAMPLE_EXTERNAL", "1").strip().lower()
    freeb_sample_external = freeb_sample_env not in ("", "0", "false", "no")
    jit_strict_update_env = os.getenv("VMEC_JAX_JIT_STRICT_UPDATE", "auto").strip().lower()
    jit_strict_update_enabled = jit_strict_update_env not in ("", "0", "false", "no", "off")
    if jit_strict_update_env == "auto":
        backend_name = _scan_backend_name()
        nrange = int(getattr(cfg, "ntor", 0)) + 1
        if bool(getattr(cfg, "lasym", False)):
            nrange = 2 * int(getattr(cfg, "ntor", 0)) + 1
        update_work = int(getattr(cfg, "ns", 0)) * int(getattr(cfg, "mpol", 0)) * int(nrange)
        try:
            cpu_work_limit = int(os.getenv("VMEC_JAX_HOST_UPDATE_CPU_WORK_LIMIT", "1000"))
        except Exception:
            cpu_work_limit = 1000
        jit_strict_update_enabled = (backend_name != "cpu") or (
            backend_name == "cpu" and (not bool(host_update_assembly)) and update_work >= cpu_work_limit
        )
    _record_setup_timing("setup_freeb_policy", _t_setup_freeb_policy)

    def _attach_freeb_diag(res: SolveVmecResidualResult) -> SolveVmecResidualResult:
        if not bool(free_boundary_enabled):
            return res
        diag_local = dict(res.diagnostics)
        if "free_boundary_external_field" not in diag_local:
            if external_field_provider_kind is not None and str(external_field_provider_kind).strip().lower() not in (
                "",
                "mgrid",
                "legacy_mgrid",
            ):
                diag_local["free_boundary_external_field"] = {
                    "enabled": True,
                    "available": False,
                    "provider_kind": str(external_field_provider_kind),
                    "reason": "direct_provider_runtime_path",
                }
            elif bool(freeb_sample_external):
                diag_local["free_boundary_external_field"] = _sample_free_boundary_external_field(
                    state=res.state,
                    static=static,
                )
            else:
                diag_local["free_boundary_external_field"] = {
                    "enabled": False,
                    "available": False,
                    "vacuum_stub": True,
                    "reason": "disabled_by_env",
                }
        return SolveVmecResidualResult(
            state=res.state,
            n_iter=int(res.n_iter),
            w_history=np.asarray(res.w_history),
            fsqr2_history=np.asarray(res.fsqr2_history),
            fsqz2_history=np.asarray(res.fsqz2_history),
            fsql2_history=np.asarray(res.fsql2_history),
            grad_rms_history=np.asarray(res.grad_rms_history),
            step_history=np.asarray(res.step_history),
            diagnostics=diag_local,
        )

    _t_setup_boundary_profiles = _setup_timer_start()
    idx00 = _mode00_index(static.modes)
    m_modes = np.asarray(
        getattr(static, "m_np", None) if getattr(static, "m_np", None) is not None else static.modes.m, dtype=int
    )
    n_modes = np.asarray(
        getattr(static, "n_np", None) if getattr(static, "n_np", None) is not None else static.modes.n, dtype=int
    )
    axis_copy_mask_np = (
        np.asarray(getattr(static, "lambda_axis_copy_mask", None), dtype=bool)
        if getattr(static, "lambda_axis_copy_mask", None) is not None
        else (m_modes == 0) & (n_modes > 0)
    )
    lambda_axis_copy_mask = jnp.asarray(axis_copy_mask_np, dtype=jnp.asarray(state0.Rcos).dtype)
    s = jnp.asarray(static.s)
    freeb_pres_scale = None
    if bool(free_boundary_enabled) and (indata is not None) and int(s.shape[0]) >= 2:
        try:
            from .profiles import eval_profiles

            hs_f = float(np.asarray(s[1] - s[0], dtype=float))
            sedge = hs_f * (float(int(s.shape[0])) - 1.5)
            prof_edge = eval_profiles(indata, np.asarray([sedge], dtype=float))
            prof_one = eval_profiles(indata, np.asarray([1.0], dtype=float))
            p_edge = float(np.asarray(prof_edge.get("pressure", np.asarray([0.0], dtype=float))).reshape(-1)[0])
            p_one = float(np.asarray(prof_one.get("pressure", np.asarray([0.0], dtype=float))).reshape(-1)[0])
            if p_edge != 0.0:
                freeb_pres_scale = p_one / p_edge
            else:
                freeb_pres_scale = 0.0
        except Exception:
            freeb_pres_scale = None
    dtype_state = jnp.asarray(state0.Rcos).dtype
    zero_precond_diag = (
        jnp.zeros((int(s.shape[0]),), dtype=dtype_state),
        jnp.zeros((int(s.shape[0]),), dtype=dtype_state),
    )
    zero_tcon = jnp.zeros((int(s.shape[0]),), dtype=dtype_state)
    constraint_active_false = jnp.asarray(False)

    # Boundary + axis recompute helpers (for VMEC-style bad-Jacobian reset).
    boundary_for_axis = (
        boundary_from_indata(indata, static.modes, apply_m1_constraint=True) if indata is not None else None
    )
    axis_reset_done = bool(resume_state is not None)
    lmove_axis = True if indata is None else bool(indata.get_bool("LMOVE_AXIS", True))
    force_axis_reset_env = os.getenv("VMEC_JAX_FORCE_AXIS_RESET_INIT", "0").strip().lower()
    force_axis_reset = force_axis_reset_env not in ("", "0", "false", "no")
    axis_reset_env = os.getenv("VMEC_JAX_AXIS_RESET_ALWAYS_3D", "0").strip().lower()
    axis_reset_always_3d = axis_reset_env not in ("", "0", "false", "no")
    axis_reset_fsq_env = os.getenv("VMEC_JAX_AXIS_RESET_FSQ_MIN", "1.0").strip()
    try:
        axis_reset_fsq_min = float(axis_reset_fsq_env) if axis_reset_fsq_env else 0.0
    except Exception:
        axis_reset_fsq_min = 0.0
    if axis_reset_fsq_min < 0.0:
        axis_reset_fsq_min = 0.0

    def _apply_vmec_lambda_axis_rules(st: VMECState) -> VMECState:
        """Enforce VMEC lambda gauge without mutating stored axis coefficients.

        VMEC applies the m=0 lambda axis-closure during real-space synthesis
        (totzsps) but does not overwrite the stored `xc` coefficients. Keep
        the state axis row intact and only enforce the (m,n)=(0,0) gauge here.
        """
        return _apply_vmec_lambda_axis_rules_to_state(
            st,
            enforce_vmec_lambda_axis=enforce_vmec_lambda_axis,
            host_update_assembly=host_update_assembly,
            idx00=idx00,
        )

    axis_reset_coeffs = None

    def _reset_axis_from_boundary(
        st: VMECState,
        *,
        k_guess=None,
        full_reset: bool = False,
        refine_axis_guess: bool = True,
    ) -> VMECState:
        nonlocal axis_reset_coeffs
        if boundary_for_axis is None:
            return st
        ntor = int(static.cfg.ntor)
        raxis_cc = np.zeros((ntor + 1,), dtype=float)
        raxis_cs = np.zeros((ntor + 1,), dtype=float)
        zaxis_cc = np.zeros((ntor + 1,), dtype=float)
        zaxis_cs = np.zeros((ntor + 1,), dtype=float)

        used_state_guess = False
        if k_guess is not None:
            try:
                raxis_cc, raxis_cs, zaxis_cc, zaxis_cs = _recompute_axis_from_state_vmec(
                    static,
                    pr1_even=k_guess.pr1_even,
                    pr1_odd=k_guess.pr1_odd,
                    pz1_even=k_guess.pz1_even,
                    pz1_odd=k_guess.pz1_odd,
                    pru_even=k_guess.pru_even,
                    pru_odd=k_guess.pru_odd,
                    pzu_even=k_guess.pzu_even,
                    pzu_odd=k_guess.pzu_odd,
                    signgs=int(signgs),
                    trig=trig,
                )
                used_state_guess = True
            except Exception:
                used_state_guess = False

        def _state_from_axis_coeffs(
            rcc: np.ndarray,
            rcs: np.ndarray,
            zcc: np.ndarray,
            zcs: np.ndarray,
            *,
            dtype,
        ) -> VMECState:
            scalars_local = dict(indata.scalars)
            scalars_local["RAXIS_CC"] = [float(v) for v in np.ravel(rcc)]
            scalars_local["RAXIS_CS"] = [float(v) for v in np.ravel(rcs)]
            scalars_local["ZAXIS_CC"] = [float(v) for v in np.ravel(zcc)]
            scalars_local["ZAXIS_CS"] = [float(v) for v in np.ravel(zcs)]
            indata_local = type(indata)(scalars=scalars_local, indexed=indata.indexed)
            return initial_guess_from_boundary(
                static,
                boundary_for_axis,
                indata_local,
                dtype=dtype,
                infer_axis_if_missing=False,
            )

        # One refinement pass on the VMEC state-based axis estimate stabilizes
        # non-axis starts where the first guess is still too far off.
        if used_state_guess and bool(refine_axis_guess):
            try:
                st_tmp = _state_from_axis_coeffs(
                    raxis_cc,
                    raxis_cs,
                    zaxis_cc,
                    zaxis_cs,
                    dtype=jnp.asarray(st.Rcos).dtype,
                )
                k_tmp, _, _, _, _, _, _, _ = _compute_forces_iter(
                    st_tmp,
                    include_edge=False,
                    zero_m1=jnp.asarray(1.0, dtype=jnp.asarray(st.Rcos).dtype),
                    constraint_precond_diag=zero_precond_diag,
                    constraint_tcon=zero_tcon,
                    constraint_precond_active=constraint_active_false,
                    constraint_tcon_active=constraint_active_false,
                    iter_idx=None,
                    iter2=1,
                )
                raxis_cc, raxis_cs, zaxis_cc, zaxis_cs = _recompute_axis_from_state_vmec(
                    static,
                    pr1_even=k_tmp.pr1_even,
                    pr1_odd=k_tmp.pr1_odd,
                    pz1_even=k_tmp.pz1_even,
                    pz1_odd=k_tmp.pz1_odd,
                    pru_even=k_tmp.pru_even,
                    pru_odd=k_tmp.pru_odd,
                    pzu_even=k_tmp.pzu_even,
                    pzu_odd=k_tmp.pzu_odd,
                    signgs=int(signgs),
                    trig=trig,
                )
            except Exception:
                pass

        if not used_state_guess:
            axis_vals = _read_axis_coeffs(indata)
            raxis_cc = np.asarray(axis_vals.get("RAXIS_CC", 0.0), dtype=float)
            zaxis_cs = np.asarray(axis_vals.get("ZAXIS_CS", 0.0), dtype=float)
            if raxis_cc.ndim == 0:
                raxis_cc = np.asarray([float(raxis_cc)], dtype=float)
            if zaxis_cs.ndim == 0:
                zaxis_cs = np.asarray([float(zaxis_cs)], dtype=float)
            if raxis_cc.size < ntor + 1:
                raxis_cc = np.pad(raxis_cc, (0, ntor + 1 - raxis_cc.size))
            if zaxis_cs.size < ntor + 1:
                zaxis_cs = np.pad(zaxis_cs, (0, ntor + 1 - zaxis_cs.size))
            raxis_cc, zaxis_cs = _recompute_axis_from_boundary(
                static,
                boundary_for_axis,
                raxis_cc=raxis_cc,
                zaxis_cs=zaxis_cs,
                signgs=int(signgs),
            )

        axis_dump_dir = os.environ.get("VMEC_JAX_DUMP_AXIS_DIR", "").strip()
        _write_axis_reset_dump(
            axis_dump_dir=axis_dump_dir,
            ns=int(static.cfg.ns),
            ntor=int(static.cfg.ntor),
            used_state_guess=bool(used_state_guess),
            raxis_cc=raxis_cc,
            raxis_cs=raxis_cs,
            zaxis_cc=zaxis_cc,
            zaxis_cs=zaxis_cs,
        )

        st_axis = _state_from_axis_coeffs(
            raxis_cc,
            raxis_cs,
            zaxis_cc,
            zaxis_cs,
            dtype=jnp.asarray(st.Rcos).dtype,
        )
        axis_reset_coeffs = (raxis_cc, raxis_cs, zaxis_cc, zaxis_cs)
        st_out = _merge_axis_reset_state(st=st, st_axis=st_axis, static=static, full_reset=full_reset)
        return _apply_vmec_lambda_axis_rules(st_out)

    def _build_wout_like_profiles(s_profile):
        flux_i = flux_profiles_from_indata(indata, s_profile, signgs=signgs)
        chipf_wout_i = jnp.asarray(flux_i.chipf)

        phips_i = jnp.asarray(flux_i.phips)
        if phips_i.shape[0] >= 1:
            phips_i = phips_i.at[0].set(0.0)

        from .boundary import boundary_from_indata

        boundary_i = boundary_from_indata(indata, static.modes)
        r00_i = (
            float(np.asarray(boundary_i.R_cos)[int(idx00)])
            if int(idx00) >= 0
            else float(np.asarray(boundary_i.R_cos)[0])
        )
        gamma_i = float(indata.get_float("GAMMA", 0.0))
        lrfp_i = bool(indata.get_bool("LRFP", False))
        chips_i = _half_mesh_from_full_mesh(chipf_wout_i) if lrfp_i else None
        mass_i = _mass_half_mesh_from_indata(
            indata=indata,
            s_full=s_profile,
            phips=phips_i,
            r00=r00_i,
            gamma=gamma_i,
            lrfp=lrfp_i,
            chips=chips_i,
        )

        pres_i = _pressure_half_mesh_from_indata(indata=indata, s_full=s_profile)
        ncurr_i = int(indata.get_int("NCURR", 0))
        icurv_i = _icurv_full_mesh_from_indata(indata=indata, s_full=s_profile, signgs=signgs)
        phipf_internal_i, chipf_internal_i, chips_eff_i = _vmec_force_flux_profiles(
            phipf=jnp.asarray(flux_i.phipf),
            chipf=chipf_wout_i,
            signgs=signgs,
            flux_is_internal=True,
        )

        wout_like_i = _WoutLikeVmecForces(
            nfp=int(static.cfg.nfp),
            mpol=int(static.cfg.mpol),
            ntor=int(static.cfg.ntor),
            lasym=bool(static.cfg.lasym),
            signgs=signgs,
            phipf=jnp.asarray(flux_i.phipf),
            phips=phips_i,
            chipf=chipf_wout_i,
            pres=pres_i,
            mass=mass_i,
            gamma=gamma_i,
            ncurr=ncurr_i,
            lcurrent=True,
            icurv=icurv_i,
            phipf_internal=phipf_internal_i,
            chipf_internal=chipf_internal_i,
            chips_eff=chips_eff_i,
        )
        return (
            flux_i,
            chipf_wout_i,
            phips_i,
            mass_i,
            pres_i,
            ncurr_i,
            icurv_i,
            phipf_internal_i,
            chipf_internal_i,
            chips_eff_i,
            wout_like_i,
        )

    _profile_numpy_patch = None
    if bool(host_update_assembly) and has_jax() and (not _tree_has_tracer(state0)):
        try:
            from .vmec_numpy_forces import _numpy_module_patch as _profile_numpy_patch
        except Exception:
            _profile_numpy_patch = None
    if _profile_numpy_patch is not None:
        with _profile_numpy_patch():
            from .vmec_numpy_forces import _wrap as _np_wrap

            s_profile = _np_wrap(np.asarray(s))
            (
                flux,
                chipf_wout,
                phips,
                mass,
                pres,
                ncurr,
                icurv,
                phipf_internal,
                chipf_internal,
                chips_eff,
                wout_like,
            ) = _build_wout_like_profiles(s_profile)
    else:
        (
            flux,
            chipf_wout,
            phips,
            mass,
            pres,
            ncurr,
            icurv,
            phipf_internal,
            chipf_internal,
            chips_eff,
            wout_like,
        ) = _build_wout_like_profiles(s)

    trig = getattr(static, "trig_vmec", None)
    if trig is None:
        trig = vmec_trig_tables(
            ntheta=int(static.cfg.ntheta),
            nzeta=int(static.cfg.nzeta),
            nfp=int(wout_like.nfp),
            mmax=int(wout_like.mpol) - 1,
            nmax=int(wout_like.ntor),
            lasym=bool(wout_like.lasym),
            dtype=jnp.asarray(state0.Rcos).dtype,
        )
    else:
        if (
            int(trig.ntheta1) != int(static.cfg.ntheta)
            or int(trig.cosnv.shape[0]) != int(static.cfg.nzeta)
            or int(trig.cosmu.shape[1]) != int(wout_like.mpol)
            or int(trig.cosnv.shape[1]) != int(wout_like.ntor) + 1
        ):
            trig = vmec_trig_tables(
                ntheta=int(static.cfg.ntheta),
                nzeta=int(static.cfg.nzeta),
                nfp=int(wout_like.nfp),
                mmax=int(wout_like.mpol) - 1,
                nmax=int(wout_like.ntor),
                lasym=bool(wout_like.lasym),
                dtype=jnp.asarray(state0.Rcos).dtype,
            )
    _record_setup_timing("setup_boundary_profiles", _t_setup_boundary_profiles)
    modes = static.modes
    # Use np.asarray for static setup data – these are closure constants captured
    # by _run_scan and converted to device arrays once at the JIT boundary.
    # Using jnp.asarray here triggers one eager XLA compilation per call (~2 ms
    # each), adding unnecessary cold-start overhead.
    m_idx = np.asarray(modes.m, dtype=np.int32)
    n_idx = np.asarray(modes.n, dtype=np.int32)
    mscale = np.asarray(trig.mscale)
    nscale = np.asarray(trig.nscale)
    idx00 = _mode00_index(static.modes)
    _traced_state0 = _tree_has_tracer(state0)
    _state_dtype = jnp.asarray(state0.Rcos).dtype if _traced_state0 else np.asarray(state0.Rcos).dtype
    lambda_update_scale_j = (
        jnp.asarray(lambda_update_scale, dtype=_state_dtype)
        if _traced_state0
        else np.asarray(lambda_update_scale, dtype=_state_dtype)
    )

    # VMEC stores Fourier coefficients in an internal (mscale/nscale) basis and
    # uses `scalxc` to represent odd-m modes in 1/sqrt(s) form. The force pipeline
    # applies `scalxc` after `tomnsps` (see `funct3d.f: gc = gc*scalxc`) so the
    # residual/preconditioner updates operate in the same internal coefficient
    # space as `VMECState`.

    if _traced_state0:
        edge_Rcos = jnp.asarray(state0.Rcos)[-1, :]
        edge_Rsin = jnp.asarray(state0.Rsin)[-1, :]
        edge_Zcos = jnp.asarray(state0.Zcos)[-1, :]
        edge_Zsin = jnp.asarray(state0.Zsin)[-1, :]
    else:
        edge_Rcos = np.asarray(state0.Rcos)[-1, :]
        edge_Rsin = np.asarray(state0.Rsin)[-1, :]
        edge_Zcos = np.asarray(state0.Zcos)[-1, :]
        edge_Zsin = np.asarray(state0.Zsin)[-1, :]

    constraint_tcon0: float | None = None
    if bool(include_constraint_force):
        constraint_tcon0 = float(indata.get_float("TCON0", 1.0))
    apply_lforbal = bool(indata.get_bool("LFORBAL", False)) if indata is not None else False

    _t_setup_cache_key_hash = _setup_timer_start()
    static_key = (
        int(static.cfg.mpol),
        int(static.cfg.ntor),
        int(static.cfg.ntheta),
        int(static.cfg.nzeta),
        int(static.cfg.nfp),
        int(static.cfg.ns),
        bool(static.cfg.lasym),
        _hash_array_bytes(static.modes.m),
        _hash_array_bytes(static.modes.n),
        _hash_array_bytes(static.grid.theta),
        _hash_array_bytes(static.grid.zeta),
    )
    wout_key = (
        int(wout_like.nfp),
        int(wout_like.mpol),
        int(wout_like.ntor),
        bool(wout_like.lasym),
        int(wout_like.signgs),
        _hash_array_bytes(wout_like.phipf),
        _hash_array_bytes(wout_like.phips),
        _hash_array_bytes(wout_like.chipf),
        _hash_array_bytes(wout_like.pres),
        _hash_array_bytes(wout_like.icurv) if getattr(wout_like, "icurv", None) is not None else None,
        float(constraint_tcon0) if constraint_tcon0 is not None else None,
    )
    edge_signature_key = _edge_signature_key(edge_Rcos, edge_Rsin, edge_Zcos, edge_Zsin)
    edge_value_key = _edge_value_key(edge_Rcos, edge_Rsin, edge_Zcos, edge_Zsin)
    _record_setup_timing("setup_cache_key_hash", _t_setup_cache_key_hash)

    def _apply_radial_tridi(a, alpha: float):
        return _radial_tridi_smooth_dirichlet(a, alpha=alpha, skip_nonpositive=True)

    def _apply_radial_tridi_batched(arrs, alpha: float):
        if alpha <= 0.0:
            return tuple(arrs)
        # Stack directly into (ns, B, ...) to avoid swapaxes.
        stack = jnp.stack(arrs, axis=1)
        smooth = _radial_tridi_smooth_dirichlet(stack, alpha=alpha)
        return tuple(smooth[:, i] for i in range(int(smooth.shape[1])))

    def _metric_surface_precond_from_bcovar(bc):
        """Approximate radial preconditioner scaling from bcovar metrics.

        Called from within JIT-compiled force computation (both CPU and GPU),
        so all operations on traced arrays must use jnp rather than np.
        The integration weights w_ang come from a static (non-traced) source
        and can be computed with plain NumPy before being passed to jnp ops.
        """
        guu = bc.guu
        r12 = bc.jac.r12
        bsubu = bc.bsubu
        bsubv = bc.bsubv
        nzeta = int(guu.shape[2])
        w_ang = jnp.asarray(vmec_wint_from_trig(trig, nzeta=nzeta), dtype=guu.dtype)
        return _metric_surface_precond_scales_jax(guu=guu, r12=r12, bsubu=bsubu, bsubv=bsubv, w_ang=w_ang)

    def _pshalf_from_s(s_arr):
        return _pshalf_from_s_np(s_arr)

    _t_setup_ptau_constants = _setup_timer_start()
    # Precompute pshalf and ohs for the JIT-accelerated ptau check.
    # These are fixed for the lifetime of this NS-stage closure. Keep the
    # cached constants tracer-safe so the residual solver can participate in a
    # JIT-compiled forward-mode Jacobian build.
    _ptau_s_is_traced = _tree_has_tracer(s)
    if _ptau_s_is_traced and has_jax():
        _ptau_s_jax = jnp.asarray(s, dtype=jnp.float64)
        if int(_ptau_s_jax.shape[0]) > 1:
            _ptau_hs_jax0 = _ptau_s_jax[1] - _ptau_s_jax[0]
        else:
            _ptau_hs_jax0 = jnp.asarray(1.0, dtype=jnp.float64)
        _ptau_ohs_scalar = None
        _ptau_pshalf_np = None
    else:
        _ptau_s_np = np.asarray(s)
        _ptau_hs = float(_ptau_s_np[1] - _ptau_s_np[0]) if int(_ptau_s_np.shape[0]) > 1 else 1.0
        _ptau_ohs_scalar = 0.0 if _ptau_hs == 0.0 else 1.0 / _ptau_hs
        _ptau_pshalf_np = _pshalf_from_s(s)
    if has_jax():
        if _ptau_s_is_traced:
            _ptau_pshalf_jax = _pshalf_from_s_jax(_ptau_s_jax, jnp.float64)
            _ptau_ohs_jax = jnp.where(
                _ptau_hs_jax0 == 0.0,
                jnp.asarray(0.0, dtype=jnp.float64),
                jnp.asarray(1.0, dtype=jnp.float64) / _ptau_hs_jax0,
            )
        else:
            _ptau_pshalf_jax = jnp.asarray(_ptau_pshalf_np, dtype=jnp.float64)
            _ptau_ohs_jax = jnp.asarray(_ptau_ohs_scalar, dtype=jnp.float64)
    _record_setup_timing("setup_ptau_constants", _t_setup_ptau_constants)

    def _ptau_minmax_from_k_host(k) -> tuple[Any | None, Any | None]:
        """Compute VMEC `ptau` min/max on the host for controller decisions."""
        # In the CPU non-scan hot path, do not call the JIT ptau helper:
        # compiling that tiny kernel shows up as avoidable cold-start overhead.
        use_host_np_ptau = bool(host_update_assembly) and (not _tree_has_tracer(k))
        return _scan_math_ptau_minmax_from_k_host(
            k,
            pshalf=_ptau_pshalf_np,
            ohs=_ptau_ohs_scalar,
            compute_jit=None if use_host_np_ptau else _ptau_compute_jit,
            pshalf_jax=None if use_host_np_ptau else (_ptau_pshalf_jax if has_jax() else None),
            ohs_jax=None if use_host_np_ptau else (_ptau_ohs_jax if has_jax() else None),
        )

    def _ptau_minmax_from_k_jax(k):
        return _scan_math_ptau_minmax_from_k_jax(k, s=s, pshalf_from_s_jax=_pshalf_from_s_jax)

    def _ptau_minmax(k):
        if has_jax():
            return _ptau_minmax_from_k_jax(k)
        return _ptau_minmax_from_k_host(k)

    def _accepted_control_ptau_arrays(k) -> tuple[Any, ...] | None:
        arrays = _scan_math_kernel_arrays_from_k(k)
        if arrays is None:
            return None
        try:
            ns = int(getattr(arrays[0], "shape", (0,))[0])
        except Exception:
            return None
        return arrays if ns >= 2 else None

    def _sm_sp_from_s(s_arr):
        return _sm_sp_from_s_np(s_arr)

    def _maybe_dump_jacobian_terms(*, k, iter_idx: int) -> None:
        _maybe_dump_jacobian_terms_record(k=k, s=s, iter_idx=iter_idx)

    def _maybe_dump_ptau(
        *,
        iter_idx: int,
        ptau_min: float,
        ptau_max: float,
        tau_min_state: float | None,
        tau_max_state: float | None,
        badjac_ptau: bool | None,
        badjac_state: bool | None,
        badjac_used: bool,
        mode: str,
        label: str,
    ) -> None:
        _runtime_maybe_dump_ptau(
            iter_idx=iter_idx,
            ptau_min=ptau_min,
            ptau_max=ptau_max,
            tau_min_state=tau_min_state,
            tau_max_state=tau_max_state,
            badjac_ptau=badjac_ptau,
            badjac_state=badjac_state,
            badjac_used=badjac_used,
            mode=mode,
            label=label,
            dump_ptau_env=os.getenv("VMEC_JAX_DUMP_PTAU", ""),
            dump_dir=os.getenv("VMEC_JAX_DUMP_DIR", ""),
        )

    def _lambda_preconditioner(bc, *, return_faclam: bool = False, return_debug: bool = False):
        lam_r0scale = float(getattr(trig, "r0scale", 1.0)) if trig is not None else 1.0
        from .preconditioner_1d_jax import lambda_preconditioner_cached

        return lambda_preconditioner_cached(
            bc=bc,
            trig=trig,
            s=s,
            cfg=cfg,
            return_faclam=return_faclam,
            return_debug=return_debug,
            r0scale=lam_r0scale,
        )

    def _rz_preconditioner_matrices_local(
        *,
        bc,
        k,
        jmax_override: int | None = None,
        use_precomputed: bool | None = None,
        use_lax_tridi: bool | None = None,
    ):
        from .preconditioner_1d_jax import rz_preconditioner_matrices

        return rz_preconditioner_matrices(
            bc=bc,
            k=k,
            trig=trig,
            s=s,
            cfg=cfg,
            jmax_override=jmax_override,
            use_precomputed=use_precomputed,
            use_lax_tridi=use_lax_tridi,
        )

    _numpy_precond_max_iter_env = os.getenv("VMEC_JAX_NUMPY_PRECOND_MAX_ITER", "240").strip()
    try:
        _numpy_precond_max_iter = int(_numpy_precond_max_iter_env)
    except Exception:
        _numpy_precond_max_iter = 240
    _use_numpy_preconditioner_apply = (
        bool(host_update_assembly)
        and int(_numpy_precond_max_iter) > 0
        and int(max_iter) <= int(_numpy_precond_max_iter)
    )

    def _rz_preconditioner_apply_local(
        *,
        frzl_in,
        mats,
        jmax,
        use_precomputed: bool | None = None,
        use_lax_tridi: bool | None = None,
    ):
        if bool(_use_numpy_preconditioner_apply) and not _tree_has_tracer(frzl_in):
            from .preconditioner_1d_jax import rz_preconditioner_apply_numpy

            return rz_preconditioner_apply_numpy(
                frzl_in=frzl_in,
                mats=mats,
                jmax=jmax,
                cfg=cfg,
                use_precomputed=use_precomputed,
            )
        from .preconditioner_1d_jax import rz_preconditioner_apply_jit

        return rz_preconditioner_apply_jit(
            frzl_in=frzl_in,
            mats=mats,
            jmax=jmax,
            cfg=cfg,
            use_precomputed=use_precomputed,
            use_lax_tridi=use_lax_tridi,
        )

    def _rz_preconditioner(frzl_in: TomnspsRZL, bc, k):
        from .preconditioner_1d_jax import rz_preconditioner

        return rz_preconditioner(
            frzl_in=frzl_in,
            bc=bc,
            k=k,
            trig=trig,
            s=s,
            cfg=cfg,
        )

    def _compute_forces(
        state: VMECState,
        *,
        include_edge: bool,
        include_edge_residual: bool | None = None,
        zero_m1: Any,
        freeb_bsqvac_half: Any | None = None,
        constraint_rcon0: Any | None = None,
        constraint_zcon0: Any | None = None,
        constraint_precond_diag: tuple[Any, Any] | None = None,
        constraint_tcon: Any | None = None,
        constraint_precond_active: Any | None = None,
        constraint_tcon_active: Any | None = None,
        iter_idx: int | None = None,
    ):
        k = vmec_forces_rz_from_wout(
            state=state,
            static=static,
            wout=wout_like,
            indata=None,
            constraint_tcon0=constraint_tcon0,
            constraint_tcon=constraint_tcon,
            constraint_precond_diag=constraint_precond_diag,
            constraint_precond_active=constraint_precond_active,
            constraint_tcon_active=constraint_tcon_active,
            constraint_rcon0=constraint_rcon0,
            constraint_zcon0=constraint_zcon0,
            freeb_bsqvac_half=freeb_bsqvac_half,
            freeb_pres_scale=freeb_pres_scale,
            use_vmec_synthesis=True,
            trig=trig,
            iter_idx=iter_idx,
        )
        if iter_idx is not None:
            _maybe_dump_bsube(bc=k.bc, static=static, iter_idx=int(iter_idx))
            _maybe_dump_bsube_terms(bc=k.bc, static=static, iter_idx=int(iter_idx))
            _maybe_dump_bsubh(bc=k.bc, static=static, iter_idx=int(iter_idx))
            _maybe_dump_bsubs(
                bc=k.bc,
                state=state,
                static=static,
                trig=trig,
                iter_idx=int(iter_idx),
                kernels=k,
            )
            _maybe_dump_lulv(bc=k.bc, static=static, iter_idx=int(iter_idx), state=state, trig=trig)
            _maybe_dump_jacobian_terms(k=k, iter_idx=int(iter_idx))
            _maybe_dump_precond_inputs(bc=k.bc, trig=trig, static=static, iter_idx=int(iter_idx), kernels=k)
            _maybe_dump_gmetric(bc=k.bc, static=static, iter_idx=int(iter_idx))
        if iter_idx is not None:
            _maybe_dump_force_kernels(k=k, static=static, iter_idx=int(iter_idx), label="raw")
        include_edge_residual = bool(include_edge if include_edge_residual is None else include_edge_residual)
        mask_pack = None
        if getattr(static, "tomnsps_masks", None) is not None:
            mask_pack = static.tomnsps_masks_edge if bool(include_edge_residual) else static.tomnsps_masks
        frzl = vmec_residual_internal_from_kernels(
            k,
            cfg_ntheta=int(static.cfg.ntheta),
            cfg_nzeta=int(static.cfg.nzeta),
            wout=wout_like,
            trig=trig,
            apply_lforbal=apply_lforbal,
            include_edge=bool(include_edge_residual),
            masks=mask_pack,
        )
        if os.getenv("VMEC_JAX_SCAN_DEBUG_FORCE", "") not in ("", "0"):
            try:
                from jax import debug as _jax_debug  # type: ignore
            except Exception:
                _jax_debug = None  # type: ignore
            if _jax_debug is not None:
                fzsc2_raw = jnp.sum(frzl.fzsc * frzl.fzsc)
                fzcs2_raw = (
                    jnp.sum(frzl.fzcs * frzl.fzcs)
                    if frzl.fzcs is not None
                    else jnp.asarray(0.0, dtype=jnp.asarray(frzl.fzsc).dtype)
                )
                _jax_debug.print(
                    "[scan-debug-raw] fzsc2_raw={fzsc:.6e} fzcs2_raw={fzcs:.6e}",
                    fzsc=fzsc2_raw,
                    fzcs=fzcs2_raw,
                )
        if os.getenv("VMEC_JAX_DUMP_HLO_FORCE_TOMNSPS", "").strip().lower() not in ("", "0", "false", "no"):
            try:

                def _tomnsps_only(k_in):
                    frzl_hlo = vmec_residual_internal_from_kernels(
                        k_in,
                        cfg_ntheta=int(static.cfg.ntheta),
                        cfg_nzeta=int(static.cfg.nzeta),
                        wout=wout_like,
                        trig=trig,
                        apply_lforbal=apply_lforbal,
                        include_edge=bool(include_edge_residual),
                        masks=mask_pack,
                    )
                    return (
                        frzl_hlo.frcc,
                        frzl_hlo.frss,
                        frzl_hlo.fzsc,
                        frzl_hlo.fzcs,
                        frzl_hlo.flsc,
                        frzl_hlo.flcs,
                    )

                _maybe_dump_hlo_kernel(
                    label="tomnsps",
                    fn=_tomnsps_only,
                    args=(k,),
                    kwargs={},
                    static=static,
                    wout_like=wout_like,
                    force=True,
                )
            except Exception:
                pass
        if iter_idx is not None:
            _maybe_dump_tomnsps(frzl=frzl, static=static, iter_idx=int(iter_idx), label="raw")
        scan_debug_force_enabled = os.getenv("VMEC_JAX_SCAN_DEBUG_FORCE", "") not in ("", "0")
        if scan_debug_force_enabled:
            if bool(apply_m1_constraints):
                frzl = vmec_apply_m1_constraints(frzl=frzl, lconm1=bool(getattr(static.cfg, "lconm1", True)))
                try:
                    from jax import debug as _jax_debug  # type: ignore
                except Exception:
                    _jax_debug = None  # type: ignore
                if _jax_debug is not None:
                    fzsc2_c = jnp.sum(frzl.fzsc * frzl.fzsc)
                    fzcs2_c = (
                        jnp.sum(frzl.fzcs * frzl.fzcs)
                        if frzl.fzcs is not None
                        else jnp.asarray(0.0, dtype=jnp.asarray(frzl.fzsc).dtype)
                    )
                    _jax_debug.print(
                        "[scan-debug-m1] fzsc2={fzsc:.6e} fzcs2={fzcs:.6e}",
                        fzsc=fzsc2_c,
                        fzcs=fzcs2_c,
                    )
            frzl = vmec_zero_m1_zforce(frzl=frzl, enabled=zero_m1)
            try:
                from jax import debug as _jax_debug  # type: ignore
            except Exception:
                _jax_debug = None  # type: ignore
            if _jax_debug is not None:
                fzsc2_z = jnp.sum(frzl.fzsc * frzl.fzsc)
                fzcs2_z = (
                    jnp.sum(frzl.fzcs * frzl.fzcs)
                    if frzl.fzcs is not None
                    else jnp.asarray(0.0, dtype=jnp.asarray(frzl.fzsc).dtype)
                )
                _jax_debug.print(
                    "[scan-debug-zero] fzsc2={fzsc:.6e} fzcs2={fzcs:.6e}",
                    fzsc=fzsc2_z,
                    fzcs=fzcs2_z,
                )
            frzl = vmec_apply_scalxc_to_tomnsps(frzl=frzl, s=s)
            frzl = _normalize_force_blocks(frzl)
        else:
            frzl = _residual_force_payload_after_m1_scalxc(
                frzl,
                s=s,
                apply_m1_constraints=bool(apply_m1_constraints),
                lconm1=bool(getattr(static.cfg, "lconm1", True)),
                zero_m1=zero_m1,
            )
        z_force_dummy = jnp.sum(frzl.fzsc)
        if frzl.fzcs is not None:
            z_force_dummy = z_force_dummy + jnp.sum(frzl.fzcs)
        if scan_debug_force_enabled:
            try:
                from jax import debug as _jax_debug  # type: ignore
            except Exception:
                _jax_debug = None  # type: ignore
            if _jax_debug is not None:
                fzsc2_s = jnp.sum(frzl.fzsc * frzl.fzsc)
                fzcs2_s = (
                    jnp.sum(frzl.fzcs * frzl.fzcs)
                    if frzl.fzcs is not None
                    else jnp.asarray(0.0, dtype=jnp.asarray(frzl.fzsc).dtype)
                )
                _jax_debug.print(
                    "[scan-debug-scalxc] fzsc2={fzsc:.6e} fzcs2={fzcs:.6e}",
                    fzsc=fzsc2_s,
                    fzcs=fzcs2_s,
                )
        if iter_idx is not None:
            _maybe_dump_gc(frzl=frzl, static=static, iter_idx=int(iter_idx), label="raw")

        # Optionally remove the LCFS contribution from the R/Z force arrays
        # before forming physical gcr2/gcz2. Keep the unmasked residual for
        # the preconditioner path (VMEC free-boundary parity).
        def _mask_edge(frzl_in: TomnspsRZL) -> TomnspsRZL:
            return _zero_edge_rz_force_blocks(frzl_in)

        frzl_full = frzl
        frzl_metric = frzl_full
        if not bool(include_edge):
            frzl_metric = _mask_edge(frzl_full)

        gcr2, gcz2, gcl2 = vmec_gcx2_from_tomnsps(
            frzl=frzl_metric,
            lconm1=bool(getattr(static.cfg, "lconm1", True)),
            apply_m1_constraints=False,
            include_edge=include_edge,
            apply_scalxc=False,
            s=s,
        )
        z_guard = jnp.where(
            jnp.isnan(z_force_dummy),
            z_force_dummy,
            jnp.asarray(0.0, dtype=jnp.asarray(z_force_dummy).dtype),
        )
        gcz2 = gcz2 + z_guard
        if iter_idx is not None:
            _maybe_dump_gcx2(
                gcr2=gcr2,
                gcz2=gcz2,
                gcl2=gcl2,
                iter_idx=int(iter_idx),
                include_edge=bool(np.asarray(include_edge)),
                ns=int(static.cfg.ns),
            )
        norms_current = vmec_force_norms_from_bcovar_dynamic(bc=k.bc, trig=trig, s=s, signgs=signgs)
        if iter_idx is not None:
            _maybe_dump_scalars(norms=norms_current, iter_idx=int(iter_idx), ns=int(static.cfg.ns))
        rz_scale, l_scale = _metric_surface_precond_from_bcovar(k.bc)
        return k, frzl_full, gcr2, gcz2, gcl2, rz_scale, l_scale, norms_current

    if os.getenv("VMEC_JAX_DUMP_HLO_DIR", "").strip():
        try:

            def _bcovar_only(st):
                from .vmec_bcovar import vmec_bcovar_half_mesh_from_wout

                return vmec_bcovar_half_mesh_from_wout(
                    state=st,
                    static=static,
                    wout=wout_like,
                    pres=None,
                    use_wout_bsup=False,
                    use_wout_bsub_for_lambda=False,
                    use_wout_bmag_for_bsq=False,
                    use_vmec_synthesis=True,
                    trig=trig,
                )

            _maybe_dump_hlo_kernel(
                label="bcovar",
                fn=_bcovar_only,
                args=(state0,),
                kwargs={},
                static=static,
                wout_like=wout_like,
            )
        except Exception:
            pass
        try:
            from .vmec_forces import vmec_forces_rz_from_wout
            from .vmec_forces import vmec_residual_internal_from_kernels

            k_hlo = vmec_forces_rz_from_wout(
                state=state0,
                static=static,
                wout=wout_like,
                indata=None,
                constraint_tcon0=constraint_tcon0,
                constraint_tcon=None,
                constraint_precond_diag=None,
                constraint_precond_active=None,
                constraint_tcon_active=None,
                use_wout_bsup=False,
                use_vmec_synthesis=True,
                trig=trig,
                iter_idx=None,
            )
            mask_pack_hlo = static.tomnsps_masks if getattr(static, "tomnsps_masks", None) is not None else None

            def _tomnsps_only(k_in):
                frzl = vmec_residual_internal_from_kernels(
                    k_in,
                    cfg_ntheta=int(static.cfg.ntheta),
                    cfg_nzeta=int(static.cfg.nzeta),
                    wout=wout_like,
                    trig=trig,
                    apply_lforbal=apply_lforbal,
                    include_edge=False,
                    masks=mask_pack_hlo,
                )
                return (frzl.frcc, frzl.frss, frzl.fzsc, frzl.fzcs, frzl.flsc, frzl.flcs)

            _maybe_dump_hlo_kernel(
                label="tomnsps",
                fn=_tomnsps_only,
                args=(k_hlo,),
                kwargs={},
                static=static,
                wout_like=wout_like,
            )
        except Exception:
            pass

    _compute_forces_impl = _compute_forces

    # NumPy hot-path: wrap _compute_forces_impl with pure-NumPy module patching.
    # Used when host_update_assembly=True to eliminate all JAX dispatch overhead.
    _compute_forces_np = None
    if bool(host_update_assembly) and has_jax():
        try:
            from .vmec_numpy_forces import compute_forces_numpy as _cfn_helper

            def _compute_forces_np(
                state: VMECState,
                *,
                include_edge: bool,
                include_edge_residual: bool | None = None,
                zero_m1: Any,
                freeb_bsqvac_half: Any | None = None,
                constraint_rcon0: Any | None = None,
                constraint_zcon0: Any | None = None,
                constraint_precond_diag: tuple[Any, Any] | None = None,
                constraint_tcon: Any | None = None,
                constraint_precond_active: Any | None = None,
                constraint_tcon_active: Any | None = None,
                iter_idx: int | None = None,
            ):
                return _cfn_helper(
                    _compute_forces_impl,
                    state,
                    include_edge=include_edge,
                    include_edge_residual=include_edge_residual,
                    zero_m1=zero_m1,
                    freeb_bsqvac_half=freeb_bsqvac_half,
                    constraint_rcon0=constraint_rcon0,
                    constraint_zcon0=constraint_zcon0,
                    constraint_precond_diag=constraint_precond_diag,
                    constraint_tcon=constraint_tcon,
                    constraint_precond_active=constraint_precond_active,
                    constraint_tcon_active=constraint_tcon_active,
                    iter_idx=iter_idx,
                )
        except Exception:
            _compute_forces_np = None

    # Pre-convert trig/wout_like/static.tomnsps_masks to pure-NumPy so that
    # indexing their array fields inside vmec_bcovar/tomnsps_rzl never triggers
    # JAX dispatch.  This must happen AFTER _compute_forces_np is set (so we
    # know the NumPy path is active) and BEFORE the iteration loop.
    #
    # Because _compute_forces is a closure that reads `trig`, `wout_like`, and
    # `static` from this enclosing scope by reference (Python cell variables),
    # reassigning them here makes the closure see the NumPy versions on every
    # subsequent call.
    #
    # static.tomnsps_masks contains TomnspsMasks with mask_even_j as a JAX
    # array; accessing it in _select_mparity triggers ~62 JAX dispatches/iter.
    # Replacing static with a shallow copy that has NumPy masks eliminates this.
    if _compute_forces_np is not None:
        try:
            import dataclasses as _dc
            import numpy as _np_host
            from .vmec_numpy_forces import _to_numpy_recursive as _tonp, _wrap as _np_wrap

            trig = _tonp(trig)
            try:
                if getattr(trig, "phase_stack", None) is not None:
                    trig = _dc.replace(
                        trig,
                        phase_stack_m=static.modes.m,
                        phase_stack_n=static.modes.n,
                    )
            except Exception:
                pass
            wout_like = _tonp(wout_like)
            # Build a replacement dict for static fields that benefit from
            # pre-conversion to _NpArray.  This eliminates JAX device→host
            # transfers that would otherwise happen on every iteration when
            # vmec_bcovar/vmec_realspace access static.s and friends.
            _repl: dict = {}
            # static.s: accessed as jnp.asarray(static.s) ~490×/run in vmec_bcovar;
            # converting to _NpArray here makes that a cheap isinstance pass.
            _s_val = getattr(static, "s", None)
            if _s_val is not None:
                try:
                    _repl["s"] = _np_wrap(_np_host.asarray(_s_val))
                except Exception:
                    pass
            # Also convert tomnsps_masks (holds JAX mask_even_j etc.)
            _np_masks = getattr(static, "tomnsps_masks", None)
            _np_masks_edge = getattr(static, "tomnsps_masks_edge", None)
            if _np_masks is not None:
                _repl["tomnsps_masks"] = _tonp(_np_masks)
                if _np_masks_edge is not None:
                    _repl["tomnsps_masks_edge"] = _tonp(_np_masks_edge)
            # Pre-convert boolean mask arrays (m_is_even, m_is_m1, etc.) to
            # float _NpArray with the state dtype so that
            #   jnp.asarray(static.m_is_even, dtype=dtype)
            # hits the fast path in _NP_MODULE.asarray and returns immediately
            # when dtype already matches, saving ~60k np.asarray calls per run.
            try:
                _state_dtype = _np_host.asarray(state0.Rcos).dtype
                for _mask_field in ("m_is_even", "m_is_odd", "m_is_m1", "m_is_odd_rest"):
                    _mval = getattr(static, _mask_field, None)
                    if _mval is not None:
                        _repl[_mask_field] = _np_wrap(_np_host.asarray(_mval, dtype=_state_dtype))
            except Exception:
                pass
            if _repl:
                static = _dc.replace(static, **_repl)
        except Exception:
            pass

    compute_cache_key = (
        "compute_forces_v1",
        static_key,
        wout_key,
        int(signgs),
        bool(apply_m1_constraints),
    )
    if jit_forces:

        def _compute_forces_nodump(
            state: VMECState,
            *,
            include_edge: bool,
            include_edge_residual: bool | None = None,
            zero_m1: Any,
            freeb_bsqvac_half: Any | None = None,
            constraint_rcon0: Any | None = None,
            constraint_zcon0: Any | None = None,
            constraint_precond_diag: tuple[Any, Any] | None = None,
            constraint_tcon: Any | None = None,
            constraint_precond_active: Any | None = None,
            constraint_tcon_active: Any | None = None,
            iter_idx: int | None = None,
        ):
            return _compute_forces_impl(
                state,
                include_edge=include_edge,
                include_edge_residual=include_edge_residual,
                zero_m1=zero_m1,
                freeb_bsqvac_half=freeb_bsqvac_half,
                constraint_rcon0=constraint_rcon0,
                constraint_zcon0=constraint_zcon0,
                constraint_precond_diag=constraint_precond_diag,
                constraint_tcon=constraint_tcon,
                constraint_precond_active=constraint_precond_active,
                constraint_tcon_active=constraint_tcon_active,
                iter_idx=None,
            )

        if differentiating_scan:
            # Do not store a jitted closure created while tracing the scan solve:
            # it can retain traced closure constants and leak them out of the
            # transformation. Primal solves still reuse the global cache.
            _compute_forces = jit(
                _compute_forces_nodump,
                static_argnames=("include_edge", "include_edge_residual"),
            )
        else:
            cached = _jit_cache_get(_COMPUTE_FORCES_CACHE, compute_cache_key)
            if cached is None:
                cached = jit(_compute_forces_nodump, static_argnames=("include_edge", "include_edge_residual"))
                cached = _jit_cache_put(
                    _COMPUTE_FORCES_CACHE,
                    compute_cache_key,
                    cached,
                    env_name="VMEC_JAX_COMPUTE_FORCES_CACHE_SIZE",
                    default=32,
                )
            _compute_forces = cached

    if bool(jit_forces) and bool(jit_precompile) and has_jax() and (jax is not None) and (_compute_forces_np is None):
        try:
            zero_m1_pre = jnp.asarray(1.0, dtype=dtype_state)
            for include_edge_flag in (False, True):
                _compute_forces.lower(
                    state0,
                    include_edge=include_edge_flag,
                    zero_m1=zero_m1_pre,
                    constraint_precond_diag=zero_precond_diag,
                    constraint_tcon=zero_tcon,
                    constraint_precond_active=constraint_active_false,
                    constraint_tcon_active=constraint_active_false,
                    iter_idx=None,
                ).compile()
        except Exception:
            pass
        need_trial_eval_precompile = bool(backtracking) or bool(reference_mode) or bool(use_direct_fallback)
        use_strict_update_precompile = (
            bool(strict_update)
            and bool(jit_strict_update_enabled)
            and (not bool(host_update_assembly))
            and (not bool(limit_dt_from_force))
            and (not bool(limit_update_rms))
            and (not bool(need_trial_eval_precompile))
            and (not _tree_has_tracer(state0))
        )
        if use_strict_update_precompile:
            try:
                velocity_shape_pre = (int(jnp.asarray(state0.Rcos).shape[0]), int(static.cfg.mpol), int(static.cfg.ntor) + 1)
                zero_update_pre = jnp.zeros(velocity_shape_pre, dtype=dtype_state)
                need_update_rms_precompile = (
                    bool(limit_update_rms)
                    or bool(track_history)
                    or bool(verbose)
                    or bool(backtracking)
                    or (bool(adjoint_trace) and adjoint_trace_mode == "full")
                )
                step_fn_pre = _strict_update_step_jit(
                    static,
                    limit_update_rms=False,
                    need_update_rms=need_update_rms_precompile,
                    divide_by_scalxc_for_update=bool(divide_by_scalxc_for_update),
                    enforce_edge=not bool(free_boundary_enabled),
                )
                scalar_pre = jnp.asarray(1.0, dtype=dtype_state)
                step_fn_pre.lower(
                    state0,
                    jnp.asarray(float(step_size), dtype=dtype_state),
                    scalar_pre,
                    scalar_pre,
                    jnp.asarray(float(step_size), dtype=dtype_state),
                    jnp.asarray(float(initial_flip_sign), dtype=dtype_state),
                    zero_update_pre,
                    zero_update_pre,
                    zero_update_pre,
                    zero_update_pre,
                    zero_update_pre,
                    zero_update_pre,
                    zero_update_pre,
                    zero_update_pre,
                    zero_update_pre,
                    zero_update_pre,
                    zero_update_pre,
                    zero_update_pre,
                    zero_update_pre,
                    zero_update_pre,
                    zero_update_pre,
                    zero_update_pre,
                    zero_update_pre,
                    zero_update_pre,
                    zero_update_pre,
                    zero_update_pre,
                    zero_update_pre,
                    zero_update_pre,
                    zero_update_pre,
                    zero_update_pre,
                    jnp.asarray(1.0e-3 if bool(reference_mode) else 5.0e-3, dtype=dtype_state),
                ).compile()
            except Exception:
                pass

    if precompile_only:
        empty = np.zeros((0,), dtype=float)
        return SolveVmecResidualResult(
            state=state0,
            n_iter=0,
            w_history=empty,
            fsqr2_history=empty,
            fsqz2_history=empty,
            fsql2_history=empty,
            grad_rms_history=empty,
            step_history=empty,
            diagnostics={"precompile_only": True},
        )

    def _iter_idx_for_dump(it: int | None) -> int | None:
        return None if jit_forces else it

    warmup_iters = int(jit_warmup_iters) if bool(jit_forces) else 0

    def _compute_forces_iter(
        state: VMECState,
        *,
        include_edge: bool,
        include_edge_residual: bool | None = None,
        zero_m1: Any,
        freeb_bsqvac_half: Any | None = None,
        constraint_rcon0: Any | None = None,
        constraint_zcon0: Any | None = None,
        constraint_precond_diag: tuple[Any, Any] | None = None,
        constraint_tcon: Any | None = None,
        constraint_precond_active: Any | None = None,
        constraint_tcon_active: Any | None = None,
        iter_idx: int | None = None,
        iter2: int | None = None,
    ):
        if warmup_iters > 0 and (iter2 is not None) and (int(iter2) <= warmup_iters):
            if has_jax():
                import jax

                with jax.disable_jit():
                    return _compute_forces_impl(
                        state,
                        include_edge=include_edge,
                        include_edge_residual=include_edge_residual,
                        zero_m1=zero_m1,
                        freeb_bsqvac_half=freeb_bsqvac_half,
                        constraint_rcon0=constraint_rcon0,
                        constraint_zcon0=constraint_zcon0,
                        constraint_precond_diag=constraint_precond_diag,
                        constraint_tcon=constraint_tcon,
                        constraint_precond_active=constraint_precond_active,
                        constraint_tcon_active=constraint_tcon_active,
                        iter_idx=iter_idx,
                    )
            return _compute_forces_impl(
                state,
                include_edge=include_edge,
                include_edge_residual=include_edge_residual,
                zero_m1=zero_m1,
                freeb_bsqvac_half=freeb_bsqvac_half,
                constraint_rcon0=constraint_rcon0,
                constraint_zcon0=constraint_zcon0,
                constraint_precond_diag=constraint_precond_diag,
                constraint_tcon=constraint_tcon,
                constraint_precond_active=constraint_precond_active,
                constraint_tcon_active=constraint_tcon_active,
                iter_idx=iter_idx,
            )
        # NumPy fast path: use pure-NumPy force computation when available.
        # This eliminates all JAX dispatch overhead from the per-iteration loop.
        if _compute_forces_np is not None:
            return _compute_forces_np(
                state,
                include_edge=include_edge,
                include_edge_residual=include_edge_residual,
                zero_m1=zero_m1,
                freeb_bsqvac_half=freeb_bsqvac_half,
                constraint_rcon0=constraint_rcon0,
                constraint_zcon0=constraint_zcon0,
                constraint_precond_diag=constraint_precond_diag,
                constraint_tcon=constraint_tcon,
                constraint_precond_active=constraint_precond_active,
                constraint_tcon_active=constraint_tcon_active,
                iter_idx=iter_idx,
            )
        return _compute_forces(
            state,
            include_edge=include_edge,
            include_edge_residual=include_edge_residual,
            zero_m1=zero_m1,
            constraint_rcon0=constraint_rcon0,
            constraint_zcon0=constraint_zcon0,
            constraint_precond_diag=constraint_precond_diag,
            constraint_tcon=constraint_tcon,
            constraint_precond_active=constraint_precond_active,
            constraint_tcon_active=constraint_tcon_active,
            iter_idx=iter_idx,
            **({"freeb_bsqvac_half": freeb_bsqvac_half} if freeb_bsqvac_half is not None else {}),
        )

    def _fsq_from_norms(norms_in, *, gcr2_in, gcz2_in, gcl2_in):
        fsqr_out = norms_in.r1 * norms_in.fnorm * gcr2_in
        fsqz_out = norms_in.r1 * norms_in.fnorm * gcz2_in
        fsql_out = norms_in.fnormL * gcl2_in
        return fsqr_out, fsqz_out, fsql_out

    _t_setup_index_constants = _setup_timer_start()
    mpol = int(static.cfg.mpol)
    ntor = int(static.cfg.ntor)
    nrange = ntor + 1
    nfp = float(static.cfg.nfp)
    ncoeff = int(jnp.asarray(state0.Rcos).shape[1])

    from .vmec_parity import signed_maps_from_modes

    signed_maps = (
        static.signed_maps if getattr(static, "signed_maps", None) is not None else signed_maps_from_modes(static.modes)
    )
    idx_pos = np.asarray(signed_maps.idx_pos, dtype=np.int32)
    idx_neg = np.asarray(signed_maps.idx_neg, dtype=np.int32)
    idx_pos_flat_np = np.asarray(signed_maps.idx_pos_flat, dtype=np.int32)
    idx_neg_flat_np = np.asarray(signed_maps.idx_neg_flat, dtype=np.int32)
    mask_pos_flat_np = np.asarray(signed_maps.mask_pos_flat)
    mask_neg_flat_np = np.asarray(signed_maps.mask_neg_flat)
    idx_pos_safe_np = np.asarray(signed_maps.idx_pos_safe_flat, dtype=np.int32)
    idx_neg_safe_np = np.asarray(signed_maps.idx_neg_safe_flat, dtype=np.int32)
    # Precompute projection matrix for _mn_*_to_signed_host: replaces np.add.at
    # with DGEMM (vals_all @ _proj_mn_signed), 9× faster per call.
    _idx_all_np = np.concatenate([idx_pos_safe_np, idx_neg_safe_np], axis=0)
    _m0_mask_np = np.asarray(signed_maps.m0_mask, dtype=bool)
    _n0_mask_np = np.asarray(signed_maps.n0_mask, dtype=bool)
    _mask_neg_bool_np = np.asarray(signed_maps.mask_neg, dtype=bool)
    _mask_pos_flat_f64 = np.asarray(signed_maps.mask_pos_flat, dtype=np.float64)
    _mask_neg_flat_f64 = np.asarray(signed_maps.mask_neg_flat, dtype=np.float64)
    if ncoeff > 0:
        _n_flat_mn = len(_idx_all_np)
        _proj_mn_signed = np.zeros((_n_flat_mn, ncoeff), dtype=np.float64)
        for _j in range(_n_flat_mn):
            _ix = int(_idx_all_np[_j])
            if 0 <= _ix < ncoeff:
                _proj_mn_signed[_j, _ix] = 1.0
        # Precompute combined projection matrices for 2-DGEMM host transforms.
        # A_cos/B_cos: map (cc_flat, ss_flat) → result in one pass, no np.where/concat.
        _n_half = _n_flat_mn // 2  # = mpol * nrange
        # m0_mask has shape (mpol, 1), n0_mask has shape (1, nrange).
        # | broadcasts to (mpol, nrange); individual reshape would give wrong size.
        _mn_bcast_shape = np.broadcast_shapes(_m0_mask_np.shape, _n0_mask_np.shape)
        _m0n0_flat_1d = (_m0_mask_np | _n0_mask_np).reshape(-1)
        _n0_flat_1d = np.broadcast_to(_n0_mask_np, _mn_bcast_shape).reshape(-1)
        _m0_flat_1d = np.broadcast_to(_m0_mask_np, _mn_bcast_shape).reshape(-1)
        _has_neg_flat = _mask_neg_flat_f64 > 0.0
        _mask_no_neg_flat = ~_has_neg_flat & ~_n0_flat_1d
        # cos: pos = 0.5*(cc+ss) unless m0|n0 → pos = cc
        _cc_pos_fac = np.where(_m0n0_flat_1d, 1.0, 0.5) * _mask_pos_flat_f64
        _ss_pos_fac = np.where(_m0n0_flat_1d, 0.0, 0.5) * _mask_pos_flat_f64
        _proj_pos = _proj_mn_signed[:_n_half]
        _proj_neg = _proj_mn_signed[_n_half:]
        _A_cos = _cc_pos_fac[:, None] * _proj_pos + 0.5 * _mask_neg_flat_f64[:, None] * _proj_neg
        _B_cos = _ss_pos_fac[:, None] * _proj_pos + (-0.5) * _mask_neg_flat_f64[:, None] * _proj_neg
        # sin: pos depends on n0/mask_no_neg/m0 category
        _sc_pos_fac = (
            np.where(
                _n0_flat_1d,
                1.0,
                np.where(_mask_no_neg_flat & _m0_flat_1d, 0.0, np.where(_mask_no_neg_flat & ~_m0_flat_1d, 1.0, 0.5)),
            )
            * _mask_pos_flat_f64
        )
        _cs_pos_fac = (
            np.where(
                _n0_flat_1d,
                0.0,
                np.where(_mask_no_neg_flat & _m0_flat_1d, -1.0, np.where(_mask_no_neg_flat & ~_m0_flat_1d, 0.0, -0.5)),
            )
            * _mask_pos_flat_f64
        )
        _A_sin = _sc_pos_fac[:, None] * _proj_pos + 0.5 * _mask_neg_flat_f64[:, None] * _proj_neg
        _B_sin = _cs_pos_fac[:, None] * _proj_pos + 0.5 * _mask_neg_flat_f64[:, None] * _proj_neg
        # Stacked matrices for single-DGEMM path: [cc, ss] @ _AB_cos
        # avoids 2 smaller DGEMMs which have worse BLAS efficiency than 1 large one.
        _AB_cos = np.vstack([_A_cos, _B_cos])  # (2*n_half, ncoeff)
        _AB_sin = np.vstack([_A_sin, _B_sin])  # (2*n_half, ncoeff)
    else:
        _proj_mn_signed = None
        _A_cos = _B_cos = _A_sin = _B_sin = None
        _AB_cos = _AB_sin = None
        _n_half = 0

    if getattr(static, "mn_idx_m", None) is not None:
        m_idx_np = np.asarray(static.mn_idx_m, dtype=np.int32)
        n_idx_np = np.asarray(static.mn_idx_n, dtype=np.int32)
        kp_idx_np = np.asarray(static.mn_idx_kp, dtype=np.int32)
        m_idx = jnp.asarray(m_idx_np)
        n_idx = jnp.asarray(n_idx_np)
        kp_idx = jnp.asarray(kp_idx_np)
        kn_idx_np = np.asarray(static.mn_idx_kn, dtype=np.int32)
        has_kn_np = np.asarray(static.mn_has_kn, dtype=bool) if static.mn_has_kn is not None else (kn_idx_np >= 0)
    else:
        m_idx_list = []
        n_idx_list = []
        kp_idx_list = []
        kn_idx_list = []
        for m_i in range(mpol):
            for n_i in range(nrange):
                kp = int(idx_pos[m_i, n_i])
                if kp < 0:
                    continue
                m_idx_list.append(m_i)
                n_idx_list.append(n_i)
                kp_idx_list.append(kp)
                kn_idx_list.append(int(idx_neg[m_i, n_i]))

        m_idx_np = np.asarray(m_idx_list, dtype=np.int32)
        n_idx_np = np.asarray(n_idx_list, dtype=np.int32)
        kp_idx_np = np.asarray(kp_idx_list, dtype=np.int32)
        m_idx = jnp.asarray(m_idx_np)
        n_idx = jnp.asarray(n_idx_np)
        kp_idx = jnp.asarray(kp_idx_np)
        kn_idx_np = np.asarray(kn_idx_list, dtype=np.int32)
        has_kn_np = kn_idx_np >= 0
    kn_idx = jnp.asarray(kn_idx_np)
    has_kn = jnp.asarray(has_kn_np)
    # NumPy index arrays for _rz_norm_np (avoid JAX dispatch on preconditioner rebuilds).
    _kp_idx_np = kp_idx_np
    _m_idx_np = m_idx_np
    _n_idx_np = n_idx_np
    _include_rcc_np = (_m_idx_np > 0) | (_n_idx_np > 0)
    _rz_norm_lthreed = bool(getattr(static.cfg, "lthreed", True))
    _rz_norm_lasym = bool(getattr(static.cfg, "lasym", False))

    def _rz_norm_np(state) -> float:
        """Pure NumPy version of _rz_norm — avoids JAX dispatch on precond rebuilds.

        Used by the host_update_assembly path to compute fnorm1 without
        blocking on XLA.  Semantically identical to _rz_norm(state).
        """
        return _geometry_rz_norm_np(
            state,
            kp_idx_np=_kp_idx_np,
            kn_idx_np=kn_idx_np,
            has_kn_np=has_kn_np,
            m_idx_np=_m_idx_np,
            n_idx_np=_n_idx_np,
            include_rcc_np=_include_rcc_np,
            lthreed=_rz_norm_lthreed,
            lasym=_rz_norm_lasym,
        )
    _record_setup_timing("setup_index_constants", _t_setup_index_constants)

    m0_mask = np.asarray(
        getattr(static, "m_is_m0", None)
        if getattr(static, "m_is_m0", None) is not None
        else (np.asarray(static.modes.m) == 0)
    )
    m0 = jnp.asarray((np.arange(mpol)[:, None] == 0))
    n0 = jnp.asarray((np.arange(nrange)[None, :] == 0))
    from .vmec_parity import _mn_cos_to_signed_cached as _mn_cos_to_signed_block
    from .vmec_parity import _mn_sin_to_signed_cached as _mn_sin_to_signed_block

    def _mn_cos_to_signed(cc, ss):
        if host_update_assembly:
            return _mn_cos_to_signed_host(cc, ss)
        cc = jnp.asarray(cc)
        ss = jnp.asarray(ss) if ss is not None else jnp.zeros_like(cc)
        return _mn_cos_to_signed_block(cc, ss, maps=signed_maps, ncoeff=ncoeff)

    def _mn_sin_to_signed(sc, cs):
        if host_update_assembly:
            return _mn_sin_to_signed_host(sc, cs)
        sc = jnp.asarray(sc)
        cs = jnp.asarray(cs) if cs is not None else jnp.zeros_like(sc)
        return _mn_sin_to_signed_block(sc, cs, maps=signed_maps, ncoeff=ncoeff)

    if has_jax():

        def _mn_sin_to_signed_batch(sc, cs):
            sc = jnp.asarray(sc)
            cs = jnp.asarray(cs) if cs is not None else jnp.zeros_like(sc)
            return jax.vmap(lambda sc_i, cs_i: _mn_sin_to_signed_block(sc_i, cs_i, maps=signed_maps, ncoeff=ncoeff))(
                sc, cs
            )
    else:

        def _mn_sin_to_signed_batch(sc, cs):
            sc = jnp.asarray(sc)
            cs = jnp.asarray(cs) if cs is not None else jnp.zeros_like(sc)
            out = [
                _mn_sin_to_signed_block(sc[i], cs[i], maps=signed_maps, ncoeff=ncoeff) for i in range(int(sc.shape[0]))
            ]
            return jnp.stack(out, axis=0)

    use_m1_pair_convert = (
        bool(getattr(static.cfg, "lthreed", True))
        and bool(getattr(static.cfg, "lconm1", True))
        and int(static.cfg.mpol) > 1
    )

    def _m1_internal_to_physical_pair(rss, zcs):
        """Convert VMEC internal m=1 (rss,zcs) pair to physical coefficients."""
        return _geometry_m1_internal_to_physical_pair(
            rss,
            zcs,
            use_m1_pair_convert=use_m1_pair_convert,
        )

    _t_setup_update_constants = _setup_timer_start()
    _state0_dtype = (
        np.asarray(state0.Rcos).dtype
        if bool(host_update_assembly) and (not _tree_has_tracer(state0.Rcos))
        else jnp.asarray(state0.Rcos).dtype
    )

    def _vmec_scalxc_from_s_np(s_in, *, mpol: int, dtype) -> np.ndarray:
        """NumPy VMEC scalxc for the non-differentiated CPU host-update path."""
        s_np = np.asarray(s_in, dtype=dtype)
        ns_local = int(s_np.shape[0])
        mpol_local = int(mpol)
        if ns_local == 0 or mpol_local <= 0:
            return np.zeros((ns_local, max(mpol_local, 0)), dtype=dtype)
        sqrts = np.sqrt(np.maximum(s_np, 0.0)).astype(dtype, copy=False)
        sqrts = np.array(sqrts, dtype=dtype, copy=True)
        sqrts[-1] = np.asarray(1.0, dtype=dtype)
        sq2 = sqrts[1] if ns_local >= 2 else np.asarray(1.0, dtype=dtype)
        scal_odd = 1.0 / np.maximum(sqrts, sq2)
        is_odd = (np.arange(mpol_local) % 2) == 1
        return np.where(is_odd[None, :], scal_odd[:, None], np.ones((ns_local, mpol_local), dtype=dtype))

    if bool(host_update_assembly) and (not _tree_has_tracer(s)):
        if bool(divide_by_scalxc_for_update):
            scalxc_mn_np = _vmec_scalxc_from_s_np(s, mpol=int(static.cfg.mpol), dtype=_state0_dtype)[:, :, None]
        else:
            scalxc_mn_np = np.ones((int(np.asarray(s).shape[0]), int(static.cfg.mpol), 1), dtype=_state0_dtype)
        # Keep a JAX value available for scan/exact helper closures, but avoid
        # constructing it through eager JAX elementwise primitives on the host path.
        scalxc_mn = scalxc_mn_np
    else:
        scalxc_mn = vmec_scalxc_from_s(s=s, mpol=int(static.cfg.mpol)).astype(jnp.asarray(state0.Rcos).dtype)[
            :, :, None
        ]
        if not bool(divide_by_scalxc_for_update):
            scalxc_mn = jnp.ones_like(scalxc_mn)
        scalxc_mn_np = (
            np.asarray(scalxc_mn, dtype=float)
            if bool(host_update_assembly) and (not _tree_has_tracer(scalxc_mn))
            else None
        )

    def _mn_cos_to_signed_host(cc, ss):
        # Single-DGEMM path: [cc, ss] @ _AB_cos (= vstack([A_cos, B_cos]))
        # Eliminates 3 np.where + old element-wise ops; keeps k=2*n_half for BLAS efficiency.
        cc_np = np.asarray(cc, dtype=float)
        ns = int(cc_np.shape[0])
        if ncoeff == 0:
            return np.zeros((ns, ncoeff), dtype=cc_np.dtype)
        if ss is None:
            return cc_np.reshape(ns, -1) @ _A_cos
        cc_ss = np.concatenate([cc_np.reshape(ns, -1), np.asarray(ss, dtype=float).reshape(ns, -1)], axis=1)
        return cc_ss @ _AB_cos

    def _mn_sin_to_signed_host(sc, cs):
        # Single-DGEMM path: [sc, cs] @ _AB_sin (= vstack([A_sin, B_sin]))
        # Eliminates 3 np.where + old element-wise ops; keeps k=2*n_half for BLAS efficiency.
        sc_np = np.asarray(sc, dtype=float)
        ns = int(sc_np.shape[0])
        if ncoeff == 0:
            return np.zeros((ns, ncoeff), dtype=sc_np.dtype)
        if cs is None:
            return sc_np.reshape(ns, -1) @ _A_sin
        sc_cs = np.concatenate([sc_np.reshape(ns, -1), np.asarray(cs, dtype=float).reshape(ns, -1)], axis=1)
        return sc_cs @ _AB_sin

    def _mn_cos_to_signed_physical(cc, ss):
        if host_update_assembly:
            cc = np.asarray(cc, dtype=float) / scalxc_mn_np
            ss = np.asarray(ss, dtype=float) / scalxc_mn_np if ss is not None else None
        else:
            cc = jnp.asarray(cc) / scalxc_mn
            ss = jnp.asarray(ss) / scalxc_mn if ss is not None else None
        return _mn_cos_to_signed(cc, ss)

    def _mn_sin_to_signed_physical(sc, cs):
        if host_update_assembly:
            sc = np.asarray(sc, dtype=float) / scalxc_mn_np
            cs = np.asarray(cs, dtype=float) / scalxc_mn_np if cs is not None else None
        else:
            sc = jnp.asarray(sc) / scalxc_mn
            cs = jnp.asarray(cs) / scalxc_mn if cs is not None else None
        return _mn_sin_to_signed(sc, cs)

    def _mn_sin_to_signed_physical_lambda(sc, cs):
        """Map lambda updates onto signed physical coefficients (VMEC scalxc)."""
        if host_update_assembly:
            sc = np.asarray(sc, dtype=float) / scalxc_mn_np
            cs = np.asarray(cs, dtype=float) / scalxc_mn_np if cs is not None else None
        else:
            sc = jnp.asarray(sc) / scalxc_mn
            cs = jnp.asarray(cs) / scalxc_mn if cs is not None else None
        return _mn_sin_to_signed(sc, cs)

    def _mn_cos_to_signed_physical_lambda(cc, ss):
        """Map asymmetric lambda updates onto signed physical coefficients (VMEC scalxc)."""
        if host_update_assembly:
            cc = np.asarray(cc, dtype=float) / scalxc_mn_np
            ss = np.asarray(ss, dtype=float) / scalxc_mn_np if ss is not None else None
        else:
            cc = jnp.asarray(cc) / scalxc_mn
            ss = jnp.asarray(ss) / scalxc_mn if ss is not None else None
        return _mn_cos_to_signed(cc, ss)

    def _mn_sin_to_signed_physical_batch(sc, cs):
        return _geometry_mn_sin_to_signed_physical_batch(
            sc,
            cs,
            scalxc_mn=scalxc_mn,
            mn_sin_to_signed_batch=_mn_sin_to_signed_batch,
        )

    def _rz_norm(state: VMECState) -> Any:
        """R/Z norm (exclude R(0,0) offset) in (m,n>=0) storage.

        This is a plain sum-of-squares over geometry Fourier coefficients in
        (m,n>=0) storage, excluding the R(0,0) offset term. For parity with the
        reference executable's norm conventions, do not apply `scalxc` here.
        """
        rpos = jnp.asarray(state.Rcos)[:, kp_idx]
        zpos = jnp.asarray(state.Zsin)[:, kp_idx]
        has_kn_mask = has_kn[None, :]
        kn_idx_safe = jnp.maximum(kn_idx, 0)
        rneg = jnp.where(has_kn_mask, jnp.asarray(state.Rcos)[:, kn_idx_safe], 0.0)
        zneg = jnp.where(has_kn_mask, jnp.asarray(state.Zsin)[:, kn_idx_safe], 0.0)
        is_m0 = (m_idx == 0)[None, :]
        rcc = rpos + jnp.where(has_kn_mask, rneg, 0.0)
        zsc = jnp.where(has_kn_mask, zpos + zneg, zpos)
        is_n0 = (n_idx == 0)[None, :]
        # VMEC m=0 uses only (rcc, zcs) for n>0; rss and zsc are canonicalized
        # to zero in internal storage.
        rss = jnp.where(is_n0 | is_m0, 0.0, jnp.where(has_kn_mask, rpos - rneg, 0.0))
        zsc = jnp.where((~is_n0) & is_m0, 0.0, zsc)
        zcs = jnp.where(is_n0, 0.0, jnp.where(has_kn_mask, zneg - zpos, -zpos))
        # Note: VMEC builds fnorm1 directly from the internal xc vector without
        # applying m=1 constraints or mscale/nscale basis normalization.

        # VMEC `bcovar_par` accumulates fnorm1 over l=2..ns (excludes axis).
        sl = slice(1, None)

        include_rcc = ((m_idx > 0) | (n_idx > 0))[None, :].astype(rcc.dtype)
        rz_norm = jnp.sum(zsc[sl] * zsc[sl]) + jnp.sum(include_rcc * (rcc[sl] * rcc[sl]))
        if bool(getattr(static.cfg, "lthreed", True)):
            rz_norm = rz_norm + jnp.sum(rss[sl] * rss[sl]) + jnp.sum(zcs[sl] * zcs[sl])
        if bool(getattr(static.cfg, "lasym", False)):
            # Asymmetric terms: include Rsin/Zcos internal components.
            rs_pos = jnp.asarray(state.Rsin)[:, kp_idx]
            zc_pos = jnp.asarray(state.Zcos)[:, kp_idx]
            rs_neg = jnp.where(has_kn_mask, jnp.asarray(state.Rsin)[:, kn_idx_safe], 0.0)
            zc_neg = jnp.where(has_kn_mask, jnp.asarray(state.Zcos)[:, kn_idx_safe], 0.0)

            # Internal sin/cos blocks from signed coefficients.
            rsc = jnp.where(has_kn_mask, rs_pos + rs_neg, jnp.where(is_n0, rs_pos, jnp.where(is_m0, 0.0, rs_pos)))
            rcs = jnp.where(has_kn_mask, rs_neg - rs_pos, jnp.where(is_n0, 0.0, jnp.where(is_m0, -rs_pos, 0.0)))

            zcc = zc_pos + jnp.where(has_kn_mask, zc_neg, 0.0)
            zss = jnp.where(is_n0 | is_m0, 0.0, jnp.where(has_kn_mask, zc_pos - zc_neg, 0.0))

            rz_norm = rz_norm + jnp.sum(rsc[sl] * rsc[sl]) + jnp.sum(rcs[sl] * rcs[sl])
            rz_norm = rz_norm + jnp.sum(zcc[sl] * zcc[sl]) + jnp.sum(zss[sl] * zss[sl])
        return rz_norm

    def _mode_diag_weights_mn(dtype):
        m = jnp.arange(mpol, dtype=jnp.float64)
        n = jnp.arange(nrange, dtype=jnp.float64) * nfp
        k2 = (m[:, None] * m[:, None]) + (n[None, :] * n[None, :])
        w = (1.0 + k2) ** (-float(mode_diag_exponent))
        return w.astype(dtype)

    def _mode_diag_weights_mn_np(dtype):
        m = np.arange(mpol, dtype=float)
        n = np.arange(nrange, dtype=float) * float(nfp)
        k2 = (m[:, None] * m[:, None]) + (n[None, :] * n[None, :])
        return ((1.0 + k2) ** (-float(mode_diag_exponent))).astype(dtype)

    # Precompute per-iteration constants once.
    if bool(host_update_assembly) and (not _tree_has_tracer(state0.Rcos)):
        w_mode_mn_np = _mode_diag_weights_mn_np(_state0_dtype)
        # JAX value is still needed by scan/exact helper closures, but the raw
        # CPU host-update path consumes the NumPy copy directly.
        w_mode_mn = jnp.asarray(w_mode_mn_np)
    else:
        w_mode_mn = _mode_diag_weights_mn(jnp.asarray(state0.Rcos).dtype)
        # NumPy copy for host-path mode-diagonal step (avoids 36 JAX dispatches/iter).
        w_mode_mn_np = (
            np.asarray(w_mode_mn) if bool(host_update_assembly) and (not _tree_has_tracer(w_mode_mn)) else None
        )
    # Precompute axis mask for _enforce_fixed_boundary_and_axis_np (avoids 7000+
    # _axis_m0_mask JAX dispatches per solve — saves ~0.5s real).
    if bool(host_update_assembly):
        if getattr(static, "m_is_m0", None) is not None:
            _precomputed_axis_mask_np = np.asarray(static.m_is_m0, dtype=_state0_dtype)
        else:
            _precomputed_axis_mask_np = (np.asarray(static.modes.m) == 0).astype(_state0_dtype)
    else:
        _precomputed_axis_mask_np = None
    # Cache JAX scalar constants used every iteration (avoids 7000+
    # jnp.asarray dispatches for zero_m1 and constraint_precond_active).
    if host_update_assembly and has_jax():
        if _tree_has_tracer(state0):
            _jnp_state_dtype = jnp.asarray(state0.Rcos).dtype
            _jnp_zero_m1_0 = jnp.asarray(0.0, dtype=_jnp_state_dtype)  # zero_m1_val=0
            _jnp_zero_m1_1 = jnp.asarray(1.0, dtype=_jnp_state_dtype)  # zero_m1_val=1
            _jnp_true_bool = jnp.asarray(True, dtype=bool)
            _jnp_false_bool = jnp.asarray(False, dtype=bool)
        else:
            _jnp_state_dtype = np.asarray(state0.Rcos).dtype
            _jnp_zero_m1_0 = np.asarray(0.0, dtype=_jnp_state_dtype)  # zero_m1_val=0
            _jnp_zero_m1_1 = np.asarray(1.0, dtype=_jnp_state_dtype)  # zero_m1_val=1
            _jnp_true_bool = np.asarray(True, dtype=bool)
            _jnp_false_bool = np.asarray(False, dtype=bool)
    else:
        _jnp_state_dtype = None
        _jnp_zero_m1_0 = _jnp_zero_m1_1 = None
        _jnp_true_bool = _jnp_false_bool = None
    # Pre-allocate zero-filled arrays for mode-diag and state-update host paths.
    # Reused every iteration instead of np.zeros_like (avoids 9+ allocations/iter).
    if host_update_assembly:
        # Shape for force arrays (ns, mpol, nrange) — used in mode-diag scaling.
        _coeff_shape_np = (int(np.asarray(state0.Rcos).shape[0]), mpol, nrange)
        _zeros_coeff_np = np.zeros(_coeff_shape_np, dtype=_state0_dtype)
        # Shape for dR/dZ/dL arrays (ns, K) — used when lasym=False for zeros.
        _zeros_dR_np = np.zeros_like(np.asarray(state0.Rcos))
    if bool(host_update_assembly) and (not _tree_has_tracer(s)) and (not _tree_has_tracer(state0.Rcos)):
        s_np = np.asarray(s)
        delta_s = (
            np.asarray(s_np[1] - s_np[0], dtype=_state0_dtype)
            if int(s_np.shape[0]) > 1
            else np.asarray(1.0, dtype=_state0_dtype)
        )
    else:
        delta_s = (
            jnp.asarray(s[1] - s[0], dtype=jnp.asarray(state0.Rcos).dtype)
            if int(jnp.asarray(s).shape[0]) > 1
            else jnp.asarray(1.0, dtype=jnp.asarray(state0.Rcos).dtype)
        )

    if bool(host_update_assembly):
        state = _enforce_fixed_boundary_and_axis_np(
            state0,
            static,
            edge_Rcos=edge_Rcos,
            edge_Rsin=edge_Rsin,
            edge_Zcos=edge_Zcos,
            edge_Zsin=edge_Zsin,
            enforce_edge=not bool(free_boundary_enabled),
            enforce_lambda_axis=True,
            idx00=idx00,
            precomputed_axis_mask=_precomputed_axis_mask_np,
        )
    else:
        state = _enforce_fixed_boundary_and_axis(
            state0,
            static,
            edge_Rcos=edge_Rcos,
            edge_Rsin=edge_Rsin,
            edge_Zcos=edge_Zcos,
            edge_Zsin=edge_Zsin,
            enforce_edge=not bool(free_boundary_enabled),
            enforce_lambda_axis=True,
            idx00=idx00,
        )
    state = _apply_vmec_lambda_axis_rules(state)

    ftol = float(indata.get_float("FTOL", 1e-13)) if ftol is None else float(ftol)
    gamma = float(indata.get_float("GAMMA", 0.0))
    if abs(gamma - 1.0) < 1e-14:
        raise ValueError("GAMMA=1 makes wp/(gamma-1) singular (VMEC objective undefined)")

    stage_prev_fsq_j = None
    if stage_prev_fsq is not None:
        try:
            stage_prev_fsq_j = jnp.asarray(float(stage_prev_fsq), dtype=dtype)
        except Exception:
            stage_prev_fsq_j = None
    _record_setup_timing("setup_update_constants", _t_setup_update_constants)

    def _run_vmec2000_scan(state_init: VMECState) -> SolveVmecResidualResult:
        scan_timing_enabled = _scan_timing_enabled(os.getenv("VMEC_JAX_TIMING", ""))
        scan_timing_stats = _new_scan_timing_stats()
        scan_total_start = time.perf_counter() if scan_timing_enabled else None

        def _scan_device_run_ready(start: float | None, value, *, cache_status: str | None = None):
            return _runtime_scan_device_run_ready(
                start=start,
                value=value,
                scan_timing_enabled=scan_timing_enabled,
                perf_counter=time.perf_counter,
                block_until_ready=jax.block_until_ready,
                tree_map=jax.tree_util.tree_map,
                record_ready=_record_scan_device_ready,
                stats=scan_timing_stats,
                cache_status=cache_status,
            )

        def _block_scan_value(value):
            return _scan_block_until_ready(
                value,
                block_until_ready=jax.block_until_ready,
                tree_map=jax.tree_util.tree_map,
            )

        _validate_vmec2000_scan_guards(
            backtracking=bool(backtracking),
            limit_dt_from_force=bool(limit_dt_from_force),
            limit_update_rms=bool(limit_update_rms),
            use_direct_fallback=bool(use_direct_fallback),
            reference_mode=bool(reference_mode),
            strict_update=bool(strict_update),
            auto_flip_force=bool(auto_flip_force),
        )

        scan_differentiated = _tree_has_tracer(state_init)
        scan_run_flags = _resolve_scan_run_flags(
            state_only=bool(state_only),
            scan_differentiated=bool(scan_differentiated),
            scan_fallback_enabled=bool(scan_fallback_enabled),
            force_chunked_scan=bool(force_chunked_scan),
        )
        state_only_scan = scan_run_flags.state_only_scan
        scan_fallback_enabled_run = scan_run_flags.scan_fallback_enabled_run
        force_chunked_scan_run = scan_run_flags.force_chunked_scan_run
        k_preconditioner_update_interval = 25
        restart_badjac_factor = 0.9
        restart_badprog_factor = 1.03
        vmec2000_fact = 1.0e4
        iter_offset0 = 0
        nstep_screen = _resolve_nstep_screen(
            indata_nstep=int(indata.get_int("NSTEP", 1)) if indata is not None else 1,
            override_env="",
        )
        tridi_precompute_env = os.getenv("VMEC_JAX_TRIDI_PRECOMPUTE", "")
        if preconditioner_use_precomputed_tridi is not None:
            tridi_precompute_env = "1" if bool(preconditioner_use_precomputed_tridi) else "0"
        tridi_solve_env = os.getenv("VMEC_JAX_TRIDI_SOLVE", "")
        if preconditioner_use_lax_tridi is not None:
            tridi_solve_env = "force" if bool(preconditioner_use_lax_tridi) else "0"
        scan_options = _vmec2000_scan_options_from_env(
            verbose=bool(verbose),
            vmec2000_control=bool(vmec2000_control),
            verbose_vmec2000_table=bool(verbose_vmec2000_table),
            light_history=bool(light_history),
            scan_minimal_default=scan_minimal_default,
            dump_any=bool(dump_any),
            fsq_total_target=fsq_total_target,
            backend_name=_scan_backend_name(),
            force_chunked_scan_run=bool(force_chunked_scan_run),
            scan_print_env=os.getenv("VMEC_JAX_SCAN_PRINT", "1"),
            scan_print_mode_env=os.getenv("VMEC_JAX_SCAN_PRINT_MODE", "debug_callback"),
            scan_print_ordered_env=os.getenv("VMEC_JAX_SCAN_PRINT_ORDERED", "0"),
            scan_print_chunked_env=os.getenv("VMEC_JAX_SCAN_PRINT_CHUNKED", "1"),
            scan_light_env=os.getenv("VMEC_JAX_SCAN_LIGHT", "0"),
            scan_minimal_env=os.getenv("VMEC_JAX_SCAN_MINIMAL", ""),
            scan_core_env=os.getenv("VMEC_JAX_SCAN_CORE", ""),
            scan_trace_env=os.getenv("VMEC_JAX_SCAN_TRACE", "0"),
            abort_scan_env=os.getenv("VMEC_JAX_SCAN_ABORT_ON_BADJAC", "0"),
            scan_precompute_env=os.getenv("VMEC_JAX_SCAN_PRECOND_PRECOMPUTE", ""),
            tridi_precompute_env=tridi_precompute_env,
            scan_lax_env=os.getenv("VMEC_JAX_SCAN_PRECOND_LAXTRIDI", ""),
            tridi_solve_env=tridi_solve_env,
            scan_restart_payload_env=os.getenv("VMEC_JAX_SCAN_RESTART_PAYLOAD", ""),
        )
        scan_options = _apply_state_only_scan_options(scan_options, state_only_scan=bool(state_only_scan))
        scan_print_env = scan_options.scan_print_env
        scan_print_mode = scan_options.scan_print_mode
        scan_print_ordered = scan_options.scan_print_ordered
        scan_light = scan_options.scan_light
        scan_minimal = scan_options.scan_minimal
        scan_collect_scalars = scan_options.scan_collect_scalars
        scan_collect_print = scan_options.scan_collect_print
        scan_core = scan_options.scan_core
        scan_trace = scan_options.scan_trace
        abort_scan_on_badjac = scan_options.abort_scan_on_badjac
        scan_use_precomputed = scan_options.scan_use_precomputed
        scan_use_lax_tridi = scan_options.scan_use_lax_tridi
        # On GPU/TPU, lax.cond executes BOTH branches unconditionally. The
        # restart payload (_restart_payload) re-runs the full vmec_bcovar +
        # force computation for the checkpoint state, doubling per-iteration
        # cost even when restarts are rare. On CPU, lax.cond branches are
        # selected at Python level (Python loop), so this overhead is avoided.
        # Default: use restart payload on CPU only; skip it on GPU/TPU.
        scan_use_restart_payload = scan_options.scan_use_restart_payload
        dump_timecontrol_scan = os.getenv("VMEC_JAX_DUMP_TIMECONTROL", "") not in ("", "0")
        scan_timecontrol_callback = None
        if dump_timecontrol_scan:
            try:
                from jax.experimental import io_callback as _io_callback

                scan_timecontrol_callback = _io_callback
            except Exception:
                dump_timecontrol_scan = False
                scan_timecontrol_callback = None
        print_in_scan = scan_options.print_in_scan
        chunked_print = scan_options.chunked_print
        _jax_debug = None
        _jax_debug_print = None
        if print_in_scan:
            try:
                from jax import debug as _jax_debug

                _jax_debug_print = _jax_debug.print
            except Exception:
                print_in_scan = False
        if scan_print_mode == "io_callback":
            try:
                from jax.experimental import io_callback as _io_callback
                scan_print_mode = _normalize_scan_print_mode(
                    scan_print_mode=scan_print_mode,
                    io_callback_available=True,
                )
            except Exception:
                scan_print_mode = _normalize_scan_print_mode(
                    scan_print_mode=scan_print_mode,
                    io_callback_available=False,
                )
                _io_callback = None  # type: ignore[assignment]
        scan_trace_ctx = None
        if scan_trace:
            try:
                from jax import profiler as _jax_profiler

                def scan_trace_ctx(label: str):  # type: ignore[misc]
                    return _jax_profiler.TraceAnnotation(label)
            except Exception:
                scan_trace = False
                scan_trace_ctx = None

        def _maybe_trace(label: str):
            if scan_trace and scan_trace_ctx is not None:
                return scan_trace_ctx(label)
            return nullcontext()

        _timecontrol_path = None
        if dump_timecontrol_scan:
            dump_dir = os.getenv("VMEC_JAX_DUMP_DIR", "").strip()
            if dump_dir:
                try:
                    _timecontrol_path = Path(dump_dir) / "time_control_trace.log"
                except Exception:
                    _timecontrol_path = None
            else:
                _timecontrol_path = None
            if _timecontrol_path is None:
                dump_timecontrol_scan = False
                scan_timecontrol_callback = None
        if resume_state is not None:
            try:
                iter_offset0 = int(resume_state.get("iter_offset", iter_offset0))
            except Exception:
                pass
        axis_reset_enabled = bool(vmec2000_control) and (not axis_reset_done) and bool(lmove_axis)
        axis_reset_repeat = False

        def _scan_hist_light(
            fsqr,
            fsqz,
            fsql,
            accepted,
            r00,
            z00,
            w_mhd,
            time_step,
            bad_jacobian,
        ):
            return (
                fsqr,
                fsqz,
                fsql,
                accepted,
                r00,
                z00,
                w_mhd,
                time_step,
                bad_jacobian,
            )

        def _scan_hist_min(fsqr, fsqz, fsql):
            return (fsqr, fsqz, fsql)

        def _should_print_vmec2000_local(iter_idx: int, max_iter_local: int) -> bool:
            return _should_print_vmec2000_row(
                iter_idx=iter_idx,
                max_iter=max_iter_local,
                nstep_screen=nstep_screen,
                verbose=bool(verbose),
                vmec2000_control=bool(vmec2000_control),
                verbose_vmec2000_table=bool(verbose_vmec2000_table),
            )

        def _print_vmec2000_row_local(
            *,
            iter_idx: int,
            fsqr: float,
            fsqz: float,
            fsql: float,
            delt0r: float,
            r00: float,
            w_mhd: float,
            z00: float | None = None,
        ) -> None:
            _print_scan_vmec2000_row(
                iter_idx=iter_idx,
                fsqr=fsqr,
                fsqz=fsqz,
                fsql=fsql,
                delt0r=delt0r,
                r00=r00,
                w_mhd=w_mhd,
                lasym=bool(cfg.lasym),
                z00=z00,
                verbose=bool(verbose),
                vmec2000_control=bool(vmec2000_control),
                verbose_vmec2000_table=bool(verbose_vmec2000_table),
            )

        def _print_axis_guess_local(raxis_cc, zaxis_cs) -> None:
            _print_scan_axis_guess(raxis_cc, zaxis_cs)

        dtype = jnp.asarray(state_init.Rcos).dtype

        def _maybe_dump_timecontrol_scan(*, cond, stage_id, iter2, iter1, fsq, fsq0, res0, res1, time_step, irst):
            if not dump_timecontrol_scan or scan_timecontrol_callback is None or _timecontrol_path is None:
                return jnp.asarray(0, dtype=jnp.int32)

            def _emit(args):
                (iter2_v, iter1_v, fsq_v, fsq0_v, res0_v, res1_v, time_step_v, irst_v, stage_id_v) = args
                _append_timecontrol_scan_trace_row(
                    _timecontrol_path,
                    stage_id=int(stage_id_v),
                    iter2=int(iter2_v),
                    iter1=int(iter1_v),
                    fsq=float(fsq_v),
                    fsq0=float(fsq0_v),
                    res0=float(res0_v),
                    res1=float(res1_v),
                    time_step=float(time_step_v),
                    irst=int(irst_v),
                )
                return np.int32(0)

            def _call(_):
                return scan_timecontrol_callback(
                    _emit,
                    jax.ShapeDtypeStruct((), jnp.int32),
                    (
                        iter2,
                        iter1,
                        fsq,
                        fsq0,
                        res0,
                        res1,
                        time_step,
                        irst,
                        stage_id,
                    ),
                    ordered=True,
                )

            return jax.lax.cond(cond, _call, lambda _: jnp.asarray(0, dtype=jnp.int32), operand=None)

        time_step0 = jnp.asarray(float(step_size), dtype=dtype)
        flip_sign0 = jnp.asarray(float(initial_flip_sign), dtype=dtype)
        fsq_total_target_j = None
        if fsq_total_target is not None:
            fsq_total_target_j = jnp.asarray(float(fsq_total_target), dtype=dtype)

        def _converged_residuals_scan(fsqr, fsqz, fsql):
            return _runtime_converged_residuals_scan_fast(
                fsqr,
                fsqz,
                fsql,
                ftol=ftol_j,
                fsq_total_target=fsq_total_target_j,
            )

        k_ndamp = 10
        scan_resume0 = _initialize_scan_resume_state(
            resume_state,
            dtype=dtype,
            velocity_shape=(int(state_init.Rcos.shape[0]), mpol, nrange),
            k_ndamp=k_ndamp,
            time_step_default=time_step0,
            flip_sign_default=flip_sign0,
            state_checkpoint_default=state_init,
        )
        time_step0 = scan_resume0.time_step
        flip_sign0 = scan_resume0.flip_sign
        inv_tau0 = scan_resume0.inv_tau
        fsq_prev0 = scan_resume0.fsq_prev
        fsq0_prev0 = scan_resume0.fsq0_prev
        res0_0 = scan_resume0.res0
        res1_0 = scan_resume0.res1
        iter1_0 = scan_resume0.iter1
        ijacob0 = scan_resume0.ijacob
        bad_resets0 = scan_resume0.bad_resets
        bad_growth0 = scan_resume0.bad_growth
        fsqz_prev0 = scan_resume0.fsqz_prev
        force_bcovar0 = scan_resume0.force_bcovar_update
        vRcc0 = scan_resume0.vRcc
        vRss0 = scan_resume0.vRss
        vZsc0 = scan_resume0.vZsc
        vZcs0 = scan_resume0.vZcs
        vLsc0 = scan_resume0.vLsc
        vLcs0 = scan_resume0.vLcs
        vRsc0 = scan_resume0.vRsc
        vRcs0 = scan_resume0.vRcs
        vZcc0 = scan_resume0.vZcc
        vZss0 = scan_resume0.vZss
        vLcc0 = scan_resume0.vLcc
        vLss0 = scan_resume0.vLss
        r00_prev0 = scan_resume0.r00_prev
        z00_prev0 = scan_resume0.z00_prev
        w_mhd_prev0 = scan_resume0.w_mhd_prev
        state_checkpoint0 = scan_resume0.state_checkpoint

        def _scale_m1_precond_rhs(frzl_in: TomnspsRZL, mats: dict[str, Any]) -> TomnspsRZL:
            return _scale_m1_precond_rhs_from_mats(
                frzl_in,
                mats,
                lconm1=getattr(cfg, "lconm1", True),
                mpol=int(cfg.mpol),
                host_update_assembly=False,
            )

        # Avoid nested JIT inside the scan by default; allow opt-in for testing.
        # Some cases benefit from a separately-jitted force kernel.
        scan_jit_env = os.getenv("VMEC_JAX_SCAN_JIT_FORCES")
        jit_forces_scan = _scan_jit_forces_enabled(env_value=scan_jit_env, jit_forces=bool(jit_forces))
        _compute_forces_scan = _compute_forces if jit_forces_scan else _compute_forces_impl
        if scan_timing_enabled and scan_total_start is not None:
            scan_timing_stats["scan_setup_s"] += time.perf_counter() - float(scan_total_start)
        t_scan_initial_force = time.perf_counter() if scan_timing_enabled else None
        with _maybe_trace("scan/compute_forces:init"):
            with _maybe_trace("scan/compute_forces:init"):
                k0, frzl0, gcr2_0, gcz2_0, gcl2_0, rz_scale0, l_scale0, norms0 = _compute_forces_scan(
                    state_init,
                    include_edge=False,
                    zero_m1=jnp.asarray(1.0, dtype=dtype),
                    constraint_precond_diag=zero_precond_diag,
                    constraint_tcon=zero_tcon,
                    constraint_precond_active=constraint_active_false,
                    constraint_tcon_active=constraint_active_false,
                    iter_idx=None,
                )
        if scan_timing_enabled and t_scan_initial_force is not None:
            try:
                if has_jax():
                    _block_scan_value((gcr2_0, gcz2_0, gcl2_0))
            except Exception:
                pass
            scan_timing_stats["scan_initial_compute_forces_s"] += time.perf_counter() - float(
                t_scan_initial_force
            )
        fsq_phys0_val = None
        try:
            fsqr0 = norms0.r1 * norms0.fnorm * gcr2_0
            fsqz0 = norms0.r1 * norms0.fnorm * gcz2_0
            fsql0 = norms0.fnormL * gcl2_0
            fsq_phys0_val = float(np.asarray(fsqr0 + fsqz0 + fsql0))
        except Exception:
            fsq_phys0_val = None
        bad_jacobian0 = False
        if axis_reset_enabled:
            axis_reset_debug = os.getenv("VMEC_JAX_AXIS_RESET_DEBUG", "").strip().lower() not in (
                "",
                "0",
                "false",
                "no",
            )
            try:
                ptau_min0, ptau_max0 = _ptau_minmax_from_k_host(k0)
            except Exception:
                ptau_min0, ptau_max0 = None, None
            bad_jacobian_ptau = None
            if (ptau_min0 is not None) and (ptau_max0 is not None):
                try:
                    min_tau_ptau0 = float(np.asarray(ptau_min0))
                    max_tau_ptau0 = float(np.asarray(ptau_max0))
                    ptau_scale0 = max(abs(min_tau_ptau0), abs(max_tau_ptau0))
                    tau_tol_ptau0 = _bad_jacobian_tau_tolerance(
                        ptau_tol=ptau_tol,
                        ptau_tol_rel=ptau_tol_rel,
                        tau_scale=ptau_scale0,
                    )
                    bad_jacobian_ptau = (min_tau_ptau0 < -tau_tol_ptau0) and (max_tau_ptau0 > tau_tol_ptau0)
                except Exception:
                    bad_jacobian_ptau = None
            bad_jacobian_state = False
            if badjac_use_state:
                try:
                    jac0 = vmec_half_mesh_jacobian_from_state(
                        state=state_init,
                        modes=static.modes,
                        trig=trig,
                        s=s,
                        lconm1=bool(getattr(static.cfg, "lconm1", True)),
                        lthreed=bool(getattr(static.cfg, "lthreed", True)),
                        mask_even=getattr(static, "m_is_even", None),
                        mask_odd=getattr(static, "m_is_odd", None),
                    )
                    tau0 = jnp.asarray(jac0.tau)
                    tau0_use = tau0[1:] if int(tau0.shape[0]) > 1 else tau0
                    min_tau_state0 = float(np.asarray(jnp.min(tau0_use)))
                    max_tau_state0 = float(np.asarray(jnp.max(tau0_use)))
                    tau_scale_state0 = max(abs(min_tau_state0), abs(max_tau_state0))
                    tau_tol_state0 = max(1.0e-12, 1.0e-2 * tau_scale_state0)
                    bad_jacobian_state = (min_tau_state0 < -tau_tol_state0) and (max_tau_state0 > tau_tol_state0)
                except Exception:
                    bad_jacobian_state = False

            axis_reset_decision = _initial_axis_reset_decision(
                bad_jacobian_ptau=bad_jacobian_ptau,
                bad_jacobian_state=bad_jacobian_state,
                badjac_use_state=badjac_use_state,
                fsq_phys=fsq_phys0_val,
                axis_reset_fsq_min=axis_reset_fsq_min,
                force_axis_reset=force_axis_reset,
                axis_reset_always_3d=axis_reset_always_3d,
                lthreed=bool(getattr(static.cfg, "lthreed", True)),
                vmec2000_control=vmec2000_control,
                lmove_axis=lmove_axis,
                axis_reset_enabled=axis_reset_enabled,
            )
            bad_jacobian0 = axis_reset_decision.bad_jacobian
            if axis_reset_debug:
                try:
                    fsq_debug_val = float("nan") if fsq_phys0_val is None else float(fsq_phys0_val)
                    print(
                        "[axis_reset] fsq0="
                        f"{fsq_debug_val:.6e} "
                        f"axis_reset_fsq_min={axis_reset_fsq_min:.3e} "
                        f"badjac_ptau={bad_jacobian_ptau} badjac_state={bad_jacobian_state} "
                        f"badjac_used={bad_jacobian0}",
                        flush=True,
                    )
                except Exception:
                    pass
        else:
            axis_reset_decision = _initial_axis_reset_decision(
                bad_jacobian_ptau=None,
                bad_jacobian_state=False,
                badjac_use_state=badjac_use_state,
                fsq_phys=fsq_phys0_val,
                axis_reset_fsq_min=axis_reset_fsq_min,
                force_axis_reset=force_axis_reset,
                axis_reset_always_3d=axis_reset_always_3d,
                lthreed=bool(getattr(static.cfg, "lthreed", True)),
                vmec2000_control=vmec2000_control,
                lmove_axis=lmove_axis,
                axis_reset_enabled=axis_reset_enabled,
            )
        force_axis_reset_init = axis_reset_decision.force_reset
        if axis_reset_decision.reset:
            if bool(verbose) and bool(vmec2000_control) and bool(verbose_vmec2000_table):
                if bad_jacobian0 or force_axis_reset_init:
                    print(" INITIAL JACOBIAN CHANGED SIGN!", flush=True)
                print(" TRYING TO IMPROVE INITIAL MAGNETIC AXIS GUESS", flush=True)
            state_init = _reset_axis_from_boundary(state_init, k_guess=k0, full_reset=False, refine_axis_guess=False)
            if bool(verbose) and bool(vmec2000_control) and bool(verbose_vmec2000_table):
                if axis_reset_coeffs is not None:
                    raxis_cc, _raxis_cs, _zaxis_cc, zaxis_cs = axis_reset_coeffs
                    _print_axis_guess_local(raxis_cc, zaxis_cs)
            ijacob0 = jnp.asarray(1, dtype=jnp.int32)
            state_checkpoint0 = state_init
            axis_reset_enabled = False
            axis_reset_repeat = True
            t_scan_axis_force = time.perf_counter() if scan_timing_enabled else None
            k0, frzl0, gcr2_0, gcz2_0, gcl2_0, rz_scale0, l_scale0, norms0 = _compute_forces_scan(
                state_init,
                include_edge=False,
                zero_m1=jnp.asarray(1.0, dtype=dtype),
                constraint_precond_diag=zero_precond_diag,
                constraint_tcon=zero_tcon,
                constraint_precond_active=constraint_active_false,
                constraint_tcon_active=constraint_active_false,
                iter_idx=None,
            )
            if scan_timing_enabled and t_scan_axis_force is not None:
                try:
                    if has_jax():
                        _block_scan_value((gcr2_0, gcz2_0, gcl2_0))
                except Exception:
                    pass
                scan_timing_stats["scan_axis_reset_compute_forces_s"] += time.perf_counter() - float(
                    t_scan_axis_force
                )
        # Axis reset handled before scan; avoid per-iteration callbacks.
        axis_reset_enabled = False
        scan_run_setup_start = time.perf_counter() if scan_timing_enabled else None
        cache_valid0 = jnp.asarray(False)
        if constraint_tcon0 is None or float(constraint_tcon0) == 0.0:
            cache_precond_diag0 = zero_precond_diag
            cache_tcon0 = zero_tcon
        else:
            from .vmec_constraints import precondn_diag_axd1_from_bcovar

            ard1_0, azd1_0 = precondn_diag_axd1_from_bcovar(
                trig=trig,
                s=s,
                bsq=k0.bc.bsq,
                r12=k0.bc.jac.r12,
                sqrtg=k0.bc.jac.sqrtg,
                ru12=k0.bc.jac.ru12,
                zu12=k0.bc.jac.zu12,
            )
            cache_precond_diag0 = (ard1_0, azd1_0)
            cache_tcon0 = jnp.asarray(k0.tcon)
        cache_norms0 = norms0
        cache_rz_scale0 = rz_scale0
        cache_l_scale0 = l_scale0
        cache_rz_norm0 = _rz_norm(state_init)
        cache_f_norm1_0 = jnp.where(cache_rz_norm0 != 0.0, 1.0 / cache_rz_norm0, jnp.asarray(float("inf"), dtype=dtype))
        from .preconditioner_1d_jax import rz_preconditioner_matrices

        cache_lam_prec0 = _lambda_preconditioner(k0.bc)
        cache_rz_mats0, _jmin0, jmax0 = rz_preconditioner_matrices(
            bc=k0.bc,
            k=k0,
            trig=trig,
            s=s,
            cfg=cfg,
            use_precomputed=bool(scan_use_precomputed),
            use_lax_tridi=bool(scan_use_lax_tridi),
        )
        # `rz_preconditioner_matrices` is JIT-compiled, so the returned jmax
        # may be a tracer when the scan solve is differentiated.  For fixed
        # grids this value is purely shape-derived.
        jmax0 = max(int(jnp.asarray(s).shape[0]) - 1, 1)
        cache_valid0 = jnp.asarray(True)

        if resume_state is not None:
            try:
                cache_valid0 = jnp.asarray(
                    bool(resume_state.get("vmec2000_cache_valid", bool(cache_valid0))), dtype=bool
                )
            except Exception:
                cache_valid0 = jnp.asarray(cache_valid0, dtype=bool)
            if "cache_precond_diag" in resume_state:
                cache_precond_diag0 = resume_state.get("cache_precond_diag", cache_precond_diag0)
            if "cache_tcon" in resume_state:
                cache_tcon0 = resume_state.get("cache_tcon", cache_tcon0)
            if "cache_norms" in resume_state:
                cache_norms0 = resume_state.get("cache_norms", cache_norms0)
            if "cache_rz_scale" in resume_state:
                cache_rz_scale0 = resume_state.get("cache_rz_scale", cache_rz_scale0)
            if "cache_l_scale" in resume_state:
                cache_l_scale0 = resume_state.get("cache_l_scale", cache_l_scale0)
            if "cache_rz_norm" in resume_state:
                try:
                    cache_rz_norm0 = jnp.asarray(resume_state.get("cache_rz_norm", cache_rz_norm0), dtype=dtype)
                except Exception:
                    pass
            if "cache_f_norm1" in resume_state:
                try:
                    cache_f_norm1_0 = jnp.asarray(resume_state.get("cache_f_norm1", cache_f_norm1_0), dtype=dtype)
                except Exception:
                    pass
            if "cache_prec_rz_mats" in resume_state:
                cache_rz_mats0 = resume_state.get("cache_prec_rz_mats", cache_rz_mats0)
            if "cache_prec_lam_prec" in resume_state:
                cache_lam_prec0 = resume_state.get("cache_prec_lam_prec", cache_lam_prec0)

        def _tree_select(cond, t_true, t_false):
            if t_true is None or t_false is None:
                return t_true if t_false is None else t_false
            if isinstance(t_true, tuple) and isinstance(t_false, tuple):
                return type(t_true)(_tree_select(cond, a, b) for a, b in zip(t_true, t_false, strict=True))
            if isinstance(t_true, list) and isinstance(t_false, list):
                return [_tree_select(cond, a, b) for a, b in zip(t_true, t_false, strict=True)]
            return jnp.where(cond, jnp.asarray(t_true), jnp.asarray(t_false))

        ftol_j = jnp.asarray(float(ftol), dtype=dtype)
        scan_fallback_iters_j = jnp.asarray(int(scan_fallback_iters), dtype=jnp.int32)
        scan_fallback_badjac_limit_j = jnp.asarray(int(scan_fallback_badjac_limit), dtype=jnp.int32)
        scan_fallback_accept_frac_j = jnp.asarray(float(scan_fallback_accept_frac), dtype=dtype)
        scan_fallback_fsq_factor_j = jnp.asarray(float(scan_fallback_fsq_factor), dtype=dtype)
        scan_fallback_fsq_abs_j = jnp.asarray(float(scan_fallback_fsq_abs), dtype=dtype)
        scan_fallback_improve_j = jnp.asarray(float(scan_fallback_improve), dtype=dtype)

        scan_jax_debug = _jax_debug
        scan_jax_debug_print = _jax_debug_print
        scan_debug_force = os.getenv("VMEC_JAX_SCAN_DEBUG_FORCE", "") not in ("", "0")
        debug_iter_env = os.getenv("VMEC_JAX_SCAN_DEBUG_ITER", "").strip()
        try:
            scan_debug_iter = int(debug_iter_env) if debug_iter_env else -1
        except Exception:
            scan_debug_iter = -1

        def _scan_step(carry: _ScanCarry, it):
            def _hold_step(carry_hold: _ScanCarry):
                return _scan_math_hold_step(
                    carry_hold,
                    dtype=dtype,
                    state_only_scan=state_only_scan,
                    scan_minimal=scan_minimal,
                    scan_light=scan_light,
                    scan_hist_min=_scan_hist_min,
                    scan_hist_light=_scan_hist_light,
                )

            def _advance_step(carry_adv: _ScanCarry):
                iter2 = jnp.asarray(it + 1, dtype=jnp.int32) + jnp.asarray(carry_adv.iter_offset, dtype=jnp.int32)
                fsq_prev_before = carry_adv.fsq_prev
                fsq0_prev_before = carry_adv.fsq0_prev
                skip_timecontrol = carry_adv.skip_timecontrol
                iter_since_restart = iter2 - carry_adv.iter1
                time_step_report = carry_adv.time_step
                # VMEC `constrain_m1`: zero gcz(m=1) on the first global
                # iteration, and again when the previous fsqz drops below the tolerance.
                zero_m1 = jnp.where(
                    (iter2 < 2) | (carry_adv.fsqz_prev < 1.0e-6),
                    jnp.asarray(1.0, dtype=dtype),
                    jnp.asarray(0.0, dtype=dtype),
                )
                prev_rz_fsq = carry_adv.fsqr_prev_phys + carry_adv.fsqz_prev_phys
                include_edge = (iter_since_restart < 50) & (prev_rz_fsq < jnp.asarray(1.0e-6, dtype=prev_rz_fsq.dtype))

                precond_age = iter2 - carry_adv.iter1
                need_periodic_precond_update = (
                    (precond_age > 0)
                    & ((precond_age % k_preconditioner_update_interval) == 0)
                )
                need_bcovar_update = (
                    (~carry_adv.cache_valid)
                    | carry_adv.force_bcovar_update
                    | need_periodic_precond_update
                )
                use_cached_precond = carry_adv.cache_valid & (~need_bcovar_update)
                constraint_precond_diag = _tree_select(
                    use_cached_precond, carry_adv.cache_precond_diag, zero_precond_diag
                )
                constraint_tcon_override = jnp.where(use_cached_precond, carry_adv.cache_tcon, zero_tcon)
                constraint_precond_active = use_cached_precond
                constraint_tcon_active = use_cached_precond

                with _maybe_trace("scan/compute_forces"):
                    k, frzl, gcr2, gcz2, gcl2, rz_scale, l_scale, norms_current = _compute_forces_scan(
                        carry_adv.state,
                        include_edge=False,
                        zero_m1=zero_m1,
                        constraint_precond_diag=constraint_precond_diag,
                        constraint_tcon=constraint_tcon_override,
                        constraint_precond_active=constraint_precond_active,
                        constraint_tcon_active=constraint_tcon_active,
                        iter_idx=None,
                    )
                norms_used = jax.lax.cond(
                    use_cached_precond,
                    lambda _: carry_adv.cache_norms,
                    lambda _: norms_current,
                    operand=None,
                )
                fsqr = norms_used.r1 * norms_used.fnorm * gcr2
                fsqz = norms_used.r1 * norms_used.fnorm * gcz2
                fsql = norms_used.fnormL * gcl2
                if scan_debug_force:
                    try:
                        from jax import debug as _jax_debug  # type: ignore
                    except Exception:
                        _jax_debug = None  # type: ignore
                    if _jax_debug is not None:

                        def _dbg(_):
                            fzsc2 = jnp.sum(frzl.fzsc * frzl.fzsc)
                            fzcs2 = (
                                jnp.sum(frzl.fzcs * frzl.fzcs)
                                if frzl.fzcs is not None
                                else jnp.asarray(0.0, dtype=fsqz.dtype)
                            )
                            fzcs_m1 = (
                                jnp.sum(frzl.fzcs[:, 1, :] * frzl.fzcs[:, 1, :])
                                if frzl.fzcs is not None and int(jnp.asarray(frzl.fzcs).shape[1]) > 1
                                else jnp.asarray(0.0, dtype=fsqz.dtype)
                            )
                            rcos_sum = jnp.sum(carry_adv.state.Rcos)
                            zsin_sum = jnp.sum(carry_adv.state.Zsin)
                            use_cached_flag = jnp.asarray(use_cached_precond, dtype=jnp.int32)
                            need_bcovar_flag = jnp.asarray(need_bcovar_update, dtype=jnp.int32)
                            if _jax_debug_print is not None:
                                _jax_debug_print(
                                    "[scan-debug] iter={i} gcr2={gcr:.6e} gcz2={gcz:.6e} fzsc2={fzsc2:.6e} fzcs2={fzcs2:.6e} fzcs_m1={fzcsm1:.6e} rcos_sum={rcsum:.6e} zsin_sum={zssum:.6e} use_cached={uc} need_bcovar={nb} fnorm={fn:.6e} r1={r1:.6e} fsqr={fsqr:.6e} fsqz={fsqz:.6e}",
                                    i=iter2,
                                    gcr=gcr2,
                                    gcz=gcz2,
                                    fzsc2=fzsc2,
                                    fzcs2=fzcs2,
                                    fzcsm1=fzcs_m1,
                                    rcsum=rcos_sum,
                                    zssum=zsin_sum,
                                    uc=use_cached_flag,
                                    nb=need_bcovar_flag,
                                    fn=norms_used.fnorm,
                                    r1=norms_used.r1,
                                    fsqr=fsqr,
                                    fsqz=fsqz,
                                )
                            return 0

                        _ = jax.lax.cond(iter2 == 1, _dbg, lambda _: 0, operand=None)
                if scan_debug_iter > 0:
                    try:
                        from jax import debug as _jax_debug_state  # type: ignore
                    except Exception:
                        _jax_debug_state = None  # type: ignore
                    if _jax_debug_state is not None:

                        def _dbg_state(_):
                            rcos_sum = jnp.sum(carry_adv.state.Rcos)
                            zsin_sum = jnp.sum(carry_adv.state.Zsin)
                            lsin_sum = jnp.sum(carry_adv.state.Lsin)
                            rcos_ck = jnp.sum(carry_adv.state_checkpoint.Rcos)
                            zsin_ck = jnp.sum(carry_adv.state_checkpoint.Zsin)
                            lsin_ck = jnp.sum(carry_adv.state_checkpoint.Lsin)
                            fsqr_dbg = norms_used.r1 * norms_used.fnorm * gcr2
                            fsqz_dbg = norms_used.r1 * norms_used.fnorm * gcz2
                            fsql_dbg = norms_used.fnormL * gcl2
                            _jax_debug_state.print(
                                "[scan-state] iter={i} rcos_sum={rc:.6e} zsin_sum={zs:.6e} lsin_sum={ls:.6e} "
                                "rcos_ck={rck:.6e} zsin_ck={zck:.6e} lsin_ck={lck:.6e} "
                                "use_cached={uc} need_bcovar={nb} gcr2={gcr:.6e} gcz2={gcz:.6e} gcl2={gcl:.6e} "
                                "fnorm={fn:.6e} r1={r1:.6e} fsqr={fsqr:.6e} fsqz={fsqz:.6e} fsql={fsql:.6e}",
                                i=iter2,
                                rc=rcos_sum,
                                zs=zsin_sum,
                                ls=lsin_sum,
                                rck=rcos_ck,
                                zck=zsin_ck,
                                lck=lsin_ck,
                                uc=jnp.asarray(use_cached_precond, dtype=jnp.int32),
                                nb=jnp.asarray(need_bcovar_update, dtype=jnp.int32),
                                gcr=gcr2,
                                gcz=gcz2,
                                gcl=gcl2,
                                fn=norms_used.fnorm,
                                r1=norms_used.r1,
                                fsqr=fsqr_dbg,
                                fsqz=fsqz_dbg,
                                fsql=fsql_dbg,
                            )
                            return 0

                        _ = jax.lax.cond(iter2 == scan_debug_iter, _dbg_state, lambda _: 0, operand=None)
                conv_now = _converged_residuals_scan(fsqr, fsqz, fsql)
                # Scalars for VMEC-style screen output (sampled on NSTEP cadence + convergence).
                sample_vmec = (iter2 <= 1) | (iter2 >= int(max_iter)) | ((iter2 % nstep_screen) == 0) | conv_now
                sample_vmec = sample_vmec & jnp.asarray(scan_collect_scalars, dtype=bool)

                def _compute_scalars(_):
                    r00_j = jnp.asarray(k.pr1_even)[0, 0, 0]
                    if bool(cfg.lasym):
                        z00_j = jnp.asarray(k.pz1_even)[0, 0, 0]
                    else:
                        z00_j = jnp.asarray(0.0, dtype=r00_j.dtype)
                    # `norms_current` already reflects the current bcovar state.
                    wb_val = jnp.asarray(norms_current.wb)
                    wp_val = jnp.asarray(norms_current.wp)
                    w_mhd = (wb_val + wp_val / (gamma - 1.0)) * jnp.asarray(float(TWOPI * TWOPI), dtype=wb_val.dtype)
                    return r00_j, z00_j, w_mhd

                def _reuse_scalars(_):
                    return carry_adv.r00_prev, carry_adv.z00_prev, carry_adv.w_mhd_prev

                r00_j, z00_j, w_mhd = jax.lax.cond(sample_vmec, _compute_scalars, _reuse_scalars, operand=None)

                def _refresh_cache(_):
                    if constraint_tcon0 is None or float(constraint_tcon0) == 0.0:
                        cache_precond_diag = zero_precond_diag
                        cache_tcon = zero_tcon
                    else:
                        from .vmec_constraints import precondn_diag_axd1_from_bcovar

                        ard1, azd1 = precondn_diag_axd1_from_bcovar(
                            trig=trig,
                            s=s,
                            bsq=k.bc.bsq,
                            r12=k.bc.jac.r12,
                            sqrtg=k.bc.jac.sqrtg,
                            ru12=k.bc.jac.ru12,
                            zu12=k.bc.jac.zu12,
                        )
                        cache_precond_diag = (ard1, azd1)
                        cache_tcon = jnp.asarray(k.tcon)
                    cache_norms = norms_used
                    cache_rz_scale = rz_scale
                    cache_l_scale = l_scale
                    cache_rz_norm = _rz_norm(carry_adv.state)
                    cache_f_norm1 = jnp.where(
                        cache_rz_norm != 0.0, 1.0 / cache_rz_norm, jnp.asarray(float("inf"), dtype=dtype)
                    )
                    from .preconditioner_1d_jax import rz_preconditioner_matrices

                    cache_lam_prec = _lambda_preconditioner(k.bc)
                    mats, _jmin, jmax = rz_preconditioner_matrices(
                        bc=k.bc,
                        k=k,
                        trig=trig,
                        s=s,
                        cfg=cfg,
                        use_precomputed=bool(scan_use_precomputed),
                        use_lax_tridi=bool(scan_use_lax_tridi),
                    )
                    # jmax is constant for fixed ns; reuse the static jmax0
                    return (
                        cache_precond_diag,
                        cache_tcon,
                        cache_norms,
                        cache_rz_scale,
                        cache_l_scale,
                        cache_rz_norm,
                        cache_f_norm1,
                        cache_lam_prec,
                        mats,
                        jnp.asarray(True),
                    )

                def _keep_cache(_):
                    return (
                        carry_adv.cache_precond_diag,
                        carry_adv.cache_tcon,
                        carry_adv.cache_norms,
                        carry_adv.cache_rz_scale,
                        carry_adv.cache_l_scale,
                        carry_adv.cache_rz_norm,
                        carry_adv.cache_f_norm1,
                        carry_adv.cache_prec_lam_prec,
                        carry_adv.cache_prec_rz_mats,
                        carry_adv.cache_valid,
                    )

                (
                    cache_precond_diag,
                    cache_tcon,
                    cache_norms,
                    cache_rz_scale,
                    cache_l_scale,
                    cache_rz_norm,
                    cache_f_norm1,
                    cache_lam_prec,
                    cache_rz_mats,
                    cache_valid,
                ) = jax.lax.cond(need_bcovar_update, _refresh_cache, _keep_cache, operand=None)

                frzl_rhs = _scale_m1_precond_rhs(frzl, cache_rz_mats)
                from .preconditioner_1d_jax import rz_preconditioner_apply

                frzl_rz = rz_preconditioner_apply(
                    frzl_in=frzl_rhs,
                    mats=cache_rz_mats,
                    jmax=jmax0,
                    cfg=cfg,
                    use_precomputed=bool(scan_use_precomputed),
                    use_lax_tridi=bool(scan_use_lax_tridi),
                )
                rz_norm = jnp.where(cache_valid, cache_rz_norm, _rz_norm(carry_adv.state))
                f_norm1 = jnp.where(
                    cache_valid,
                    cache_f_norm1,
                    jnp.where(rz_norm != 0.0, 1.0 / rz_norm, jnp.asarray(float("inf"), dtype=dtype)),
                )
                current_payload_pre = _current_scan_payload(
                    frzl_rz=frzl_rz,
                    cache_lam_prec=cache_lam_prec,
                    w_mode_mn=w_mode_mn,
                    lambda_update_scale_j=lambda_update_scale_j,
                    apply_lambda_update_scale=(lambda_update_scale != 1.0),
                    fsqr=fsqr,
                    fsqz=fsqz,
                    fsql=fsql,
                    f_norm1=f_norm1,
                    delta_s=delta_s,
                    s=s,
                    lconm1=bool(getattr(static.cfg, "lconm1", True)),
                    cache_precond_diag=cache_precond_diag,
                    cache_tcon=cache_tcon,
                    cache_norms=cache_norms,
                    cache_rz_scale=cache_rz_scale,
                    cache_l_scale=cache_l_scale,
                    cache_rz_norm=cache_rz_norm,
                    cache_f_norm1=cache_f_norm1,
                    cache_rz_mats=cache_rz_mats,
                    cache_valid=cache_valid,
                    lambda_fsq1_optional_source=frzl,
                )
                (
                    frcc_u,
                    frss_u,
                    fzsc_u,
                    fzcs_u,
                    flsc_u,
                    flcs_u,
                    frsc_u,
                    frcs_u,
                    fzcc_u,
                    fzss_u,
                    flcc_u,
                    flss_u,
                ) = current_payload_pre.blocks
                fsqr1 = current_payload_pre.fsqr1
                fsqz1 = current_payload_pre.fsqz1
                fsql1 = current_payload_pre.fsql1
                fsq1 = fsqr1 + fsqz1 + fsql1

                fsq0 = fsqr + fsqz + fsql
                if bool(vmec2000_control):
                    ptau_min, ptau_max = _ptau_minmax(k)
                    if (ptau_min is None) or (ptau_max is None):
                        min_tau_ptau = jnp.asarray(jnp.nan, dtype=dtype)
                        max_tau_ptau = jnp.asarray(jnp.nan, dtype=dtype)
                        bad_jacobian_ptau = jnp.asarray(False)
                    else:
                        min_tau_ptau = jnp.asarray(ptau_min)
                        max_tau_ptau = jnp.asarray(ptau_max)
                        tau_tol_ptau = jnp.asarray(abs(ptau_tol), dtype=dtype)
                        bad_jacobian_ptau = (min_tau_ptau < -tau_tol_ptau) & (max_tau_ptau > tau_tol_ptau)
                    ptau_valid = jnp.isfinite(min_tau_ptau) & jnp.isfinite(max_tau_ptau)
                    state_probe = (
                        badjac_state_probe
                        & (jnp.asarray(badjac_initial_state_probe_iters, dtype=jnp.int32) > 0)
                        & (iter2 <= jnp.asarray(badjac_initial_state_probe_iters, dtype=jnp.int32))
                    )
                    need_state_jac = (
                        badjac_use_state | dump_ptau_state | state_probe | (~ptau_valid) | bad_jacobian_ptau
                    )

                    def _state_jacobian():
                        jac_scan = vmec_half_mesh_jacobian_from_state(
                            state=carry_adv.state,
                            modes=static.modes,
                            trig=trig,
                            s=s,
                            lconm1=bool(getattr(static.cfg, "lconm1", True)),
                            lthreed=bool(getattr(static.cfg, "lthreed", True)),
                            mask_even=getattr(static, "m_is_even", None),
                            mask_odd=getattr(static, "m_is_odd", None),
                        )
                        tau = jnp.asarray(jac_scan.tau)
                        tau_decision = _scan_math_state_jacobian(
                            tau,
                            vmec2000_control=True,
                            ptau_tol=ptau_tol,
                            relative_tol=1.0e-2,
                        )
                        return tau_decision.bad_jacobian, tau_decision.min_tau, tau_decision.max_tau

                    def _ptau_only():
                        return jnp.asarray(False), jnp.asarray(jnp.nan, dtype=dtype), jnp.asarray(jnp.nan, dtype=dtype)

                    badjac_state, min_tau_state, max_tau_state = jax.lax.cond(
                        need_state_jac, _state_jacobian, _ptau_only
                    )
                    badjac_ptau = bad_jacobian_ptau
                    min_tau = jnp.where(
                        badjac_use_state,
                        min_tau_state,
                        min_tau_ptau,
                    )
                    max_tau = jnp.where(
                        badjac_use_state,
                        max_tau_state,
                        max_tau_ptau,
                    )
                    bad_jacobian = jnp.where(
                        badjac_use_state,
                        badjac_state,
                        badjac_ptau,
                    )
                elif not use_apply_payload_fusion:
                    use_state_jac = os.getenv("VMEC_JAX_SCAN_JAC_FROM_STATE", "0").strip().lower() not in (
                        "",
                        "0",
                        "false",
                        "no",
                    )
                    if use_state_jac:
                        jac_scan = vmec_half_mesh_jacobian_from_state(
                            state=carry_adv.state,
                            modes=static.modes,
                            trig=trig,
                            s=s,
                            lconm1=bool(getattr(static.cfg, "lconm1", True)),
                            lthreed=bool(getattr(static.cfg, "lthreed", True)),
                            mask_even=getattr(static, "m_is_even", None),
                            mask_odd=getattr(static, "m_is_odd", None),
                        )
                        tau = jnp.asarray(jac_scan.tau)
                    else:
                        tau = jnp.asarray(k.bc.jac.tau)
                    tau_decision = _scan_math_state_jacobian(
                        tau,
                        vmec2000_control=False,
                        ptau_tol=ptau_tol,
                    )
                    bad_jacobian = tau_decision.bad_jacobian
                    min_tau = tau_decision.min_tau
                    max_tau = tau_decision.max_tau
                    badjac_ptau = jnp.asarray(False)
                    badjac_state = bad_jacobian
                    min_tau_state = min_tau
                    max_tau_state = max_tau
                    min_tau_ptau = min_tau
                    max_tau_ptau = max_tau
                if os.getenv("VMEC_JAX_SCAN_IGNORE_BADJAC", "") not in ("", "0"):
                    bad_jacobian = jnp.asarray(False)
                # Axis reset handled before entering the scan loop.

                fsq_phys = fsq0
                if bool(vmec2000_control):
                    fsq_phys = jnp.where(
                        bad_jacobian & (iter2 > carry_adv.iter1),
                        fsq0_prev_before,
                        fsq_phys,
                    )
                if bool(vmec2000_control):
                    # VMEC2000 TimeStepControl uses the *previous* preconditioned residual.
                    fsq = carry_adv.fsq_prev
                    fsq_res = fsq
                elif not use_apply_payload_fusion:
                    fsq_res = jnp.where(jnp.asarray(reference_mode), fsq_phys, fsq1)
                    fsq = fsq_res
                init_mask = (iter2 == carry_adv.iter1) | (carry_adv.res0 < 0.0) | (carry_adv.res1 < 0.0)

                tc_scalars = scan_time_control_scalars(
                    skip_timecontrol=skip_timecontrol,
                    init_mask=init_mask,
                    fsq=fsq,
                    fsq_res=fsq_res,
                    fsq_phys=fsq_phys,
                    fsq1=fsq1,
                    fsq_prev_before=fsq_prev_before,
                    res0_prev=carry_adv.res0,
                    res1_prev=carry_adv.res1,
                    bad_jacobian=bad_jacobian,
                    vmec2000_control=bool(vmec2000_control),
                )
                res0 = tc_scalars.res0
                res1 = tc_scalars.res1
                checkpoint_mask = tc_scalars.checkpoint_mask

                # Important for scan performance: avoid element-wise `where`
                # over full state arrays. VMEC checkpointing is a discrete
                # choice, so keep it as conditionals driven by scalar masks.
                state_checkpoint_init = jax.lax.cond(
                    (~skip_timecontrol) & init_mask,
                    lambda _: carry_adv.state,
                    lambda _: carry_adv.state_checkpoint,
                    operand=None,
                )
                state_checkpoint = jax.lax.cond(
                    checkpoint_mask,
                    lambda _: carry_adv.state,
                    lambda _: state_checkpoint_init,
                    operand=None,
                )
                fsqr_checkpoint = jnp.where(checkpoint_mask, fsqr, carry_adv.fsqr_checkpoint)
                fsqz_checkpoint = jnp.where(checkpoint_mask, fsqz, carry_adv.fsqz_checkpoint)
                fsql_checkpoint = jnp.where(checkpoint_mask, fsql, carry_adv.fsql_checkpoint)
                fsqr1_checkpoint = jnp.where(checkpoint_mask, fsqr1, carry_adv.fsqr1_checkpoint)
                fsqz1_checkpoint = jnp.where(checkpoint_mask, fsqz1, carry_adv.fsqz1_checkpoint)
                fsql1_checkpoint = jnp.where(checkpoint_mask, fsql1, carry_adv.fsql1_checkpoint)

                if dump_timecontrol_scan:
                    _maybe_dump_timecontrol_scan(
                        cond=(~skip_timecontrol) & init_mask,
                        stage_id=jnp.asarray(0, dtype=jnp.int32),
                        iter2=iter2,
                        iter1=carry_adv.iter1,
                        fsq=fsq,
                        fsq0=fsq_phys,
                        res0=res0,
                        res1=res1,
                        time_step=carry_adv.time_step,
                        irst=jnp.asarray(1, dtype=jnp.int32),
                    )
                    _maybe_dump_timecontrol_scan(
                        cond=~skip_timecontrol,
                        stage_id=jnp.asarray(1, dtype=jnp.int32),
                        iter2=iter2,
                        iter1=carry_adv.iter1,
                        fsq=fsq,
                        fsq0=fsq_phys,
                        res0=res0,
                        res1=res1,
                        time_step=carry_adv.time_step,
                        irst=jnp.asarray(1, dtype=jnp.int32),
                    )
                    _maybe_dump_timecontrol_scan(
                        cond=checkpoint_mask,
                        stage_id=jnp.asarray(2, dtype=jnp.int32),
                        iter2=iter2,
                        iter1=carry_adv.iter1,
                        fsq=fsq,
                        fsq0=fsq_phys,
                        res0=res0,
                        res1=res1,
                        time_step=carry_adv.time_step,
                        irst=jnp.asarray(1, dtype=jnp.int32),
                    )

                restart_decision = scan_restart_decision(
                    skip_timecontrol=skip_timecontrol,
                    iter2=iter2,
                    iter1=carry_adv.iter1,
                    fsq=fsq,
                    fsq_phys=fsq_phys,
                    res0=res0,
                    res1=res1,
                    bad_jacobian=bad_jacobian,
                    fsqr=fsqr,
                    fsqz=fsqz,
                    vmec2000_fact=vmec2000_fact,
                    use_restart_triggers=bool(use_restart_triggers),
                    vmecpp_restart=bool(vmecpp_restart),
                    k_preconditioner_update_interval=k_preconditioner_update_interval,
                    stage_prev_fsq=stage_prev_fsq_j,
                    stage_transition_factor=stage_transition_factor,
                    vmec2000_control=bool(vmec2000_control),
                )
                stage_spike = restart_decision.stage_spike
                do_restart = restart_decision.do_restart
                restart_reason = restart_decision.restart_reason
                if dump_timecontrol_scan:
                    _maybe_dump_timecontrol_scan(
                        cond=do_restart,
                        stage_id=jnp.asarray(3, dtype=jnp.int32),
                        iter2=iter2,
                        iter1=carry_adv.iter1,
                        fsq=fsq,
                        fsq0=fsq_phys,
                        res0=res0,
                        res1=res1,
                        time_step=carry_adv.time_step,
                        irst=restart_decision.irst_restart,
                    )

                def _restart_updates(_):
                    return _scan_math_restart_updates(
                        carry_adv=carry_adv,
                        state_checkpoint=state_checkpoint,
                        fsq_prev_before=fsq_prev_before,
                        iter2=iter2,
                        restart_reason=restart_reason,
                        vmec2000_control=bool(vmec2000_control),
                        restart_badjac_factor=restart_badjac_factor,
                        restart_badprog_factor=restart_badprog_factor,
                        stage_transition_scale=stage_transition_scale,
                        step_size=step_size,
                        k_ndamp=k_ndamp,
                        dtype=dtype,
                        scan_restart_transition_fn=scan_restart_transition,
                    )

                def _no_restart_updates(_):
                    return _scan_math_no_restart_updates(carry_adv)

                (
                    state_post,
                    time_step_post,
                    inv_tau_post,
                    fsq_prev_post,
                    vRcc_post,
                    vRss_post,
                    vZsc_post,
                    vZcs_post,
                    vLsc_post,
                    vLcs_post,
                    vRsc_post,
                    vRcs_post,
                    vZcc_post,
                    vZss_post,
                    vLcc_post,
                    vLss_post,
                    iter_offset_post,
                    iter1_post,
                    ijacob_post,
                    bad_resets_post,
                    bad_growth_post,
                    force_bcovar_post,
                ) = jax.lax.cond(do_restart, _restart_updates, _no_restart_updates, operand=None)

                fsq0_prev_post = jnp.where(do_restart, fsq0_prev_before, fsq_phys)

                stage_post_update = scan_stage_spike_post_update(
                    time_step=time_step_post,
                    inv_tau=inv_tau_post,
                    velocity_blocks=(
                        vRcc_post,
                        vRss_post,
                        vZsc_post,
                        vZcs_post,
                        vLsc_post,
                        vLcs_post,
                        vRsc_post,
                        vRcs_post,
                        vZcc_post,
                        vZss_post,
                        vLcc_post,
                        vLss_post,
                    ),
                    iter1=iter1_post,
                    iter2=iter2,
                    stage_spike=stage_spike,
                    stage_prev_fsq=stage_prev_fsq_j,
                    stage_transition_scale=stage_transition_scale,
                    k_ndamp=k_ndamp,
                    dtype=dtype,
                )
                time_step_post = stage_post_update.time_step
                inv_tau_post = stage_post_update.inv_tau
                (
                    vRcc_post,
                    vRss_post,
                    vZsc_post,
                    vZcs_post,
                    vLsc_post,
                    vLcs_post,
                    vRsc_post,
                    vRcs_post,
                    vZcc_post,
                    vZss_post,
                    vLcc_post,
                    vLss_post,
                ) = stage_post_update.velocity_blocks
                iter1_post = stage_post_update.iter1

                def _restart_payload(_):
                    with _maybe_trace("scan/compute_forces:restart"):
                        k_r, frzl_r, gcr2_r, gcz2_r, gcl2_r, rz_scale_r, l_scale_r, norms_current_r = (
                            _compute_forces_scan(
                                state_post,
                                include_edge=False,
                                zero_m1=zero_m1,
                                constraint_precond_diag=zero_precond_diag,
                                constraint_tcon=zero_tcon,
                                constraint_precond_active=constraint_active_false,
                                constraint_tcon_active=constraint_active_false,
                                iter_idx=None,
                            )
                        )
                    norms_used_r = norms_current_r
                    fsqr_r = norms_used_r.r1 * norms_used_r.fnorm * gcr2_r
                    fsqz_r = norms_used_r.r1 * norms_used_r.fnorm * gcz2_r
                    fsql_r = norms_used_r.fnormL * gcl2_r

                    rz_norm_r = _rz_norm(state_post)
                    f_norm1_r = jnp.where(rz_norm_r != 0.0, 1.0 / rz_norm_r, jnp.asarray(float("inf"), dtype=dtype))

                    if constraint_tcon0 is None or float(constraint_tcon0) == 0.0:
                        cache_precond_diag_r = zero_precond_diag
                        cache_tcon_r = zero_tcon
                    else:
                        from .vmec_constraints import precondn_diag_axd1_from_bcovar

                        ard1_r, azd1_r = precondn_diag_axd1_from_bcovar(
                            trig=trig,
                            s=s,
                            bsq=k_r.bc.bsq,
                            r12=k_r.bc.jac.r12,
                            sqrtg=k_r.bc.jac.sqrtg,
                            ru12=k_r.bc.jac.ru12,
                            zu12=k_r.bc.jac.zu12,
                        )
                        cache_precond_diag_r = (ard1_r, azd1_r)
                        cache_tcon_r = jnp.asarray(k_r.tcon)
                    cache_norms_r = norms_used_r
                    cache_rz_scale_r = rz_scale_r
                    cache_l_scale_r = l_scale_r
                    cache_rz_norm_r = rz_norm_r
                    cache_f_norm1_r = f_norm1_r
                    cache_lam_prec_r = _lambda_preconditioner(k_r.bc)
                    from .preconditioner_1d_jax import rz_preconditioner_matrices

                    mats_r, _jmin, jmax_r = rz_preconditioner_matrices(
                        bc=k_r.bc,
                        k=k_r,
                        trig=trig,
                        s=s,
                        cfg=cfg,
                        use_precomputed=bool(scan_use_precomputed),
                        use_lax_tridi=bool(scan_use_lax_tridi),
                    )
                    cache_valid_r = jnp.asarray(True)

                    frzl_rhs_r = _scale_m1_precond_rhs(frzl_r, mats_r)
                    from .preconditioner_1d_jax import rz_preconditioner_apply

                    frzl_rz_r = rz_preconditioner_apply(
                        frzl_in=frzl_rhs_r,
                        mats=mats_r,
                        jmax=jmax0,
                        cfg=cfg,
                        use_precomputed=bool(scan_use_precomputed),
                        use_lax_tridi=bool(scan_use_lax_tridi),
                    )
                    return _restart_scan_payload(
                        frzl_rz=frzl_rz_r,
                        cache_lam_prec=cache_lam_prec_r,
                        w_mode_mn=w_mode_mn,
                        lambda_update_scale_j=lambda_update_scale_j,
                        apply_lambda_update_scale=(lambda_update_scale != 1.0),
                        fsqr=fsqr_r,
                        fsqz=fsqz_r,
                        fsql=fsql_r,
                        f_norm1=f_norm1_r,
                        delta_s=delta_s,
                        s=s,
                        lconm1=bool(getattr(static.cfg, "lconm1", True)),
                        cache_precond_diag=cache_precond_diag_r,
                        cache_tcon=cache_tcon_r,
                        cache_norms=cache_norms_r,
                        cache_rz_scale=cache_rz_scale_r,
                        cache_l_scale=cache_l_scale_r,
                        cache_rz_norm=cache_rz_norm_r,
                        cache_f_norm1=cache_f_norm1_r,
                        cache_rz_mats=mats_r,
                        cache_valid=cache_valid_r,
                    )

                def _current_payload(_):
                    return current_payload_pre

                payload_use = _select_scan_force_payload(
                    do_restart=do_restart,
                    use_restart_payload=bool(scan_use_restart_payload),
                    restart_payload_fn=_restart_payload,
                    current_payload_fn=_current_payload,
                    cond=jax.lax.cond,
                )
                (
                    frcc_u_use,
                    frss_u_use,
                    fzsc_u_use,
                    fzcs_u_use,
                    flsc_u_use,
                    flcs_u_use,
                    frsc_u_use,
                    frcs_u_use,
                    fzcc_u_use,
                    fzss_u_use,
                    flcc_u_use,
                    flss_u_use,
                ) = payload_use.blocks
                fsqr_use = payload_use.fsqr
                fsqz_use = payload_use.fsqz
                fsql_use = payload_use.fsql
                fsqr1_use = payload_use.fsqr1
                fsqz1_use = payload_use.fsqz1
                fsql1_use = payload_use.fsql1
                cache_precond_diag_use = payload_use.cache_precond_diag
                cache_tcon_use = payload_use.cache_tcon
                cache_norms_use = payload_use.cache_norms
                cache_rz_scale_use = payload_use.cache_rz_scale
                cache_l_scale_use = payload_use.cache_l_scale
                cache_rz_norm_use = payload_use.cache_rz_norm
                cache_f_norm1_use = payload_use.cache_f_norm1
                cache_rz_mats_use = payload_use.cache_rz_mats
                cache_lam_prec_use = payload_use.cache_lam_prec
                cache_valid_use = payload_use.cache_valid

                frcc_u = frcc_u_use
                frss_u = frss_u_use
                fzsc_u = fzsc_u_use
                fzcs_u = fzcs_u_use
                flsc_u = flsc_u_use
                flcs_u = flcs_u_use
                frsc_u = frsc_u_use
                frcs_u = frcs_u_use
                fzcc_u = fzcc_u_use
                fzss_u = fzss_u_use
                flcc_u = flcc_u_use
                flss_u = flss_u_use
                fsqr = fsqr_use
                fsqz = fsqz_use
                fsql = fsql_use
                fsqr1 = fsqr1_use
                fsqz1 = fsqz1_use
                fsql1 = fsql1_use
                fsq1 = fsqr1 + fsqz1 + fsql1

                def _accept_step(_):
                    inv_tau_reset = jnp.full((k_ndamp,), jnp.asarray(0.15, dtype=dtype) / time_step_post)
                    invtau_num = jnp.where(
                        fsq1 == 0.0,
                        0.0,
                        jnp.minimum(jnp.abs(jnp.log(fsq1 / jnp.maximum(fsq_prev_post, 1.0e-30))), 0.15),
                    )
                    inv_tau_next = jnp.concatenate([inv_tau_post[1:], invtau_num[None] / time_step_post], axis=0)
                    inv_tau = jnp.where(iter2 == iter1_post, inv_tau_reset, inv_tau_next)
                    fsq_prev = fsq1
                    otav = jnp.sum(inv_tau) / float(k_ndamp)
                    dtau = time_step_post * otav / 2.0
                    b1 = 1.0 - dtau
                    fac = 1.0 / (1.0 + dtau)
                    force_scale = time_step_post
                    vRcc = fac * (b1 * vRcc_post + force_scale * (flip_sign0 * frcc_u))
                    vRss = fac * (b1 * vRss_post + force_scale * (flip_sign0 * frss_u))
                    vRsc = fac * (b1 * vRsc_post + force_scale * (flip_sign0 * frsc_u))
                    vRcs = fac * (b1 * vRcs_post + force_scale * (flip_sign0 * frcs_u))
                    vZsc = fac * (b1 * vZsc_post + force_scale * (flip_sign0 * fzsc_u))
                    vZcs = fac * (b1 * vZcs_post + force_scale * (flip_sign0 * fzcs_u))
                    vZcc = fac * (b1 * vZcc_post + force_scale * (flip_sign0 * fzcc_u))
                    vZss = fac * (b1 * vZss_post + force_scale * (flip_sign0 * fzss_u))
                    vLsc = fac * (b1 * vLsc_post + force_scale * (flip_sign0 * flsc_u))
                    vLcs = fac * (b1 * vLcs_post + force_scale * (flip_sign0 * flcs_u))
                    vLcc = fac * (b1 * vLcc_post + force_scale * (flip_sign0 * flcc_u))
                    vLss = fac * (b1 * vLss_post + force_scale * (flip_sign0 * flss_u))
                    dR = time_step_post * _mn_cos_to_signed_physical(vRcc, vRss)
                    dZ = time_step_post * _mn_sin_to_signed_physical(vZsc, vZcs)
                    dL = time_step_post * _mn_sin_to_signed_physical_lambda(vLsc, vLcs)
                    if bool(cfg.lasym):
                        dR_sin = time_step_post * _mn_sin_to_signed_physical(vRsc, vRcs)
                        dZ_cos = time_step_post * _mn_cos_to_signed_physical(vZcc, vZss)
                        dL_cos = time_step_post * _mn_cos_to_signed_physical_lambda(vLcc, vLss)
                    else:
                        dR_sin = jnp.zeros_like(dR)
                        dZ_cos = jnp.zeros_like(dR)
                        dL_cos = jnp.zeros_like(dR)
                    state_new = VMECState(
                        layout=state_post.layout,
                        Rcos=jnp.asarray(state_post.Rcos) + dR,
                        Rsin=jnp.asarray(state_post.Rsin) + dR_sin,
                        Zcos=jnp.asarray(state_post.Zcos) + dZ_cos,
                        Zsin=jnp.asarray(state_post.Zsin) + dZ,
                        Lcos=jnp.asarray(state_post.Lcos) + dL_cos,
                        Lsin=jnp.asarray(state_post.Lsin) + dL,
                    )
                    state_new = _enforce_fixed_boundary_and_axis(
                        state_new,
                        static,
                        edge_Rcos=carry.edge_Rcos,
                        edge_Rsin=carry.edge_Rsin,
                        edge_Zcos=carry.edge_Zcos,
                        edge_Zsin=carry.edge_Zsin,
                        enforce_edge=not bool(free_boundary_enabled),
                        enforce_lambda_axis=True,
                        idx00=idx00,
                    )
                    state_new = _apply_vmec_lambda_axis_rules(state_new)
                    return _ScanStepFields(
                        state=state_new,
                        vRcc=vRcc,
                        vRss=vRss,
                        vZsc=vZsc,
                        vZcs=vZcs,
                        vLsc=vLsc,
                        vLcs=vLcs,
                        vRsc=vRsc,
                        vRcs=vRcs,
                        vZcc=vZcc,
                        vZss=vZss,
                        vLcc=vLcc,
                        vLss=vLss,
                        inv_tau=inv_tau,
                        fsq_prev=fsq_prev,
                    )

                def _reject_step(_):
                    return _ScanStepFields(
                        state=state_post,
                        vRcc=vRcc_post,
                        vRss=vRss_post,
                        vZsc=vZsc_post,
                        vZcs=vZcs_post,
                        vLsc=vLsc_post,
                        vLcs=vLcs_post,
                        vRsc=vRsc_post,
                        vRcs=vRcs_post,
                        vZcc=vZcc_post,
                        vZss=vZss_post,
                        vLcc=vLcc_post,
                        vLss=vLss_post,
                        inv_tau=inv_tau_post,
                        fsq_prev=fsq_prev_post,
                    )

                step_fields = _select_scan_step_fields(
                    vmec2000_control=bool(vmec2000_control),
                    do_restart=do_restart,
                    accept_step_fn=_accept_step,
                    reject_step_fn=_reject_step,
                    cond=jax.lax.cond,
                )
                (
                    state_new,
                    vRcc_new,
                    vRss_new,
                    vZsc_new,
                    vZcs_new,
                    vLsc_new,
                    vLcs_new,
                    vRsc_new,
                    vRcs_new,
                    vZcc_new,
                    vZss_new,
                    vLcc_new,
                    vLss_new,
                    inv_tau_new,
                    fsq_prev_new,
                ) = step_fields
                fsq0_prev_new = fsq0_prev_post

                fsqr_out = fsqr
                fsqz_out = fsqz
                fsql_out = fsql
                fsqr1_out = fsqr1
                fsqz1_out = fsqz1
                fsql1_out = fsql1

                restart_effective = do_restart if not bool(vmec2000_control) else jnp.asarray(False)
                fsqz_prev = jnp.where(restart_effective, carry_adv.fsqz_prev, fsqz_out)

                accepted = jnp.logical_not(do_restart)
                accepted_count_new = jnp.where(
                    jnp.asarray(scan_core),
                    carry_adv.accepted_count,
                    carry_adv.accepted_count + jnp.asarray(accepted, dtype=jnp.int32),
                )
                fallback_active = carry_adv.fallback_active
                probe_update = scan_fallback_probe_update(
                    enabled=scan_fallback_enabled_run,
                    scan_core=bool(scan_core),
                    probe_count=carry_adv.probe_count,
                    probe_bad_jac=carry_adv.probe_bad_jac,
                    probe_accept=carry_adv.probe_accept,
                    probe_fsq_start=carry_adv.probe_fsq_start,
                    probe_fsq_min=carry_adv.probe_fsq_min,
                    probe_fsq_max=carry_adv.probe_fsq_max,
                    fallback_active=fallback_active,
                    abort_scan=carry_adv.abort_scan,
                    fsq_phys=fsq_phys,
                    fsq1=fsq1,
                    bad_jacobian=bad_jacobian,
                    accepted=accepted,
                    abort_scan_on_badjac=abort_scan_on_badjac,
                    fallback_iters=scan_fallback_iters_j,
                    badjac_limit=scan_fallback_badjac_limit_j,
                    accept_frac=scan_fallback_accept_frac_j,
                    fsq_factor=scan_fallback_fsq_factor_j,
                    fsq_abs=scan_fallback_fsq_abs_j,
                    improve=scan_fallback_improve_j,
                    dtype=dtype,
                )
                probe_count_new = probe_update.probe_count
                probe_bad_jac_new = probe_update.probe_bad_jac
                probe_accept_new = probe_update.probe_accept
                probe_fsq_start_new = probe_update.probe_fsq_start
                probe_fsq_min_new = probe_update.probe_fsq_min
                probe_fsq_max_new = probe_update.probe_fsq_max
                abort_scan_new = probe_update.abort_scan

                if bool(vmec2000_control):
                    cache_valid_out = cache_valid_use
                else:
                    cache_valid_out = jnp.where(do_restart, jnp.asarray(False), cache_valid)
                # VMEC prints the updated time-step (post TimeStepControl/restart),
                # so report the post-update value on this iteration.
                time_step_report = time_step_post
                if print_in_scan:

                    def _do_print(_):
                        if _scan_print_uses_debug_print(
                            scan_print_mode=scan_print_mode,
                            debug_print_fn=scan_jax_debug_print,
                        ):
                            scan_jax_debug_print(
                                "{i:5d}{fsqr:10.2E}{fsqz:10.2E}{fsql:10.2E}{r00:11.3E}{dt:10.2E}{w:12.4E}",
                                i=iter2,
                                fsqr=fsqr,
                                fsqz=fsqz,
                                fsql=fsql,
                                r00=r00_j,
                                dt=time_step_report,
                                w=w_mhd,
                                ordered=bool(scan_print_ordered),
                            )
                        elif _scan_print_uses_debug_callback(
                            scan_print_mode=scan_print_mode,
                            debug_module=scan_jax_debug,
                        ):

                            def _cb(i, fsqr_v, fsqz_v, fsql_v, r00_v, dt_v, w_v):
                                print(
                                    f"{int(i):5d}"
                                    f"{float(fsqr_v):10.2E}{float(fsqz_v):10.2E}{float(fsql_v):10.2E}"
                                    f"{float(r00_v):11.3E}{float(dt_v):10.2E}{float(w_v):12.4E}",
                                    flush=True,
                                )
                                return None

                            scan_jax_debug.callback(
                                _cb,
                                iter2,
                                fsqr,
                                fsqz,
                                fsql,
                                r00_j,
                                time_step_report,
                                w_mhd,
                                ordered=bool(scan_print_ordered),
                            )
                        elif _scan_print_uses_io_callback(
                            scan_print_mode=scan_print_mode,
                            io_callback_fn=_io_callback,
                        ):

                            def _cb_io(i, fsqr_v, fsqz_v, fsql_v, r00_v, dt_v, w_v):
                                print(
                                    f"{int(i):5d}"
                                    f"{float(fsqr_v):10.2E}{float(fsqz_v):10.2E}{float(fsql_v):10.2E}"
                                    f"{float(r00_v):11.3E}{float(dt_v):10.2E}{float(w_v):12.4E}",
                                    flush=True,
                                )
                                return ()

                            _io_callback(  # type: ignore[misc]
                                _cb_io,
                                None,
                                iter2,
                                fsqr,
                                fsqz,
                                fsql,
                                r00_j,
                                time_step_report,
                                w_mhd,
                                ordered=bool(scan_print_ordered),
                            )
                        return 0

                    _ = jax.lax.cond(sample_vmec, _do_print, lambda _: 0, operand=None)
                new_carry = _ScanCarry(
                    state=state_new,
                    time_step=time_step_post,
                    inv_tau=inv_tau_new,
                    fsq_prev=fsq_prev_new,
                    fsq0_prev=fsq0_prev_new,
                    accepted_count=accepted_count_new,
                    abort_scan=abort_scan_new,
                    skip_timecontrol=jnp.asarray(False) if bool(vmec2000_control) else jnp.asarray(do_restart),
                    vRcc=vRcc_new,
                    vRss=vRss_new,
                    vZsc=vZsc_new,
                    vZcs=vZcs_new,
                    vLsc=vLsc_new,
                    vLcs=vLcs_new,
                    vRsc=vRsc_new,
                    vRcs=vRcs_new,
                    vZcc=vZcc_new,
                    vZss=vZss_new,
                    vLcc=vLcc_new,
                    vLss=vLss_new,
                    flip_sign=flip_sign0,
                    iter_offset=iter_offset_post,
                    iter1=iter1_post,
                    res0=res0,
                    res1=res1,
                    state_checkpoint=state_checkpoint,
                    cache_valid=cache_valid_out,
                    cache_precond_diag=cache_precond_diag,
                    cache_tcon=cache_tcon,
                    cache_norms=cache_norms,
                    cache_rz_scale=cache_rz_scale,
                    cache_l_scale=cache_l_scale,
                    cache_rz_norm=cache_rz_norm,
                    cache_f_norm1=cache_f_norm1,
                    cache_prec_rz_mats=cache_rz_mats,
                    cache_prec_lam_prec=cache_lam_prec,
                    force_bcovar_update=jnp.asarray(False) if bool(vmec2000_control) else force_bcovar_post,
                    ijacob=ijacob_post,
                    bad_resets=bad_resets_post,
                    bad_growth=bad_growth_post,
                    fsqz_prev=fsqz_prev,
                    r00_prev=r00_j,
                    z00_prev=z00_j,
                    w_mhd_prev=w_mhd,
                    converged=carry_adv.converged | conv_now,
                    probe_count=probe_count_new,
                    probe_bad_jac=probe_bad_jac_new,
                    probe_accept=probe_accept_new,
                    probe_fsq_min=probe_fsq_min_new,
                    probe_fsq_max=probe_fsq_max_new,
                    probe_fsq_start=probe_fsq_start_new,
                    fallback_active=fallback_active,
                    fsqr_prev_phys=jnp.where(restart_effective, carry_adv.fsqr_prev_phys, fsqr_out),
                    fsqz_prev_phys=jnp.where(restart_effective, carry_adv.fsqz_prev_phys, fsqz_out),
                    fsql_prev_phys=jnp.where(restart_effective, carry_adv.fsql_prev_phys, fsql_out),
                    fsqr1_prev=jnp.where(restart_effective, carry_adv.fsqr1_prev, fsqr1_out),
                    fsqz1_prev=jnp.where(restart_effective, carry_adv.fsqz1_prev, fsqz1_out),
                    fsql1_prev=jnp.where(restart_effective, carry_adv.fsql1_prev, fsql1_out),
                    fsqr_checkpoint=fsqr_checkpoint,
                    fsqz_checkpoint=fsqz_checkpoint,
                    fsql_checkpoint=fsql_checkpoint,
                    fsqr1_checkpoint=fsqr1_checkpoint,
                    fsqz1_checkpoint=fsqz1_checkpoint,
                    fsql1_checkpoint=fsql1_checkpoint,
                    edge_Rcos=carry.edge_Rcos,
                    edge_Rsin=carry.edge_Rsin,
                    edge_Zcos=carry.edge_Zcos,
                    edge_Zsin=carry.edge_Zsin,
                )
                if state_only_scan:
                    return new_carry, ()
                if scan_minimal:
                    return new_carry, _scan_hist_min(fsqr_out, fsqz_out, fsql_out)
                if scan_light:
                    return new_carry, _scan_hist_light(
                        fsqr_out,
                        fsqz_out,
                        fsql_out,
                        accepted,
                        r00_j,
                        z00_j,
                        w_mhd,
                        time_step_report,
                        bad_jacobian,
                    )
                return new_carry, (
                    fsqr_out,
                    fsqz_out,
                    fsql_out,
                    fsqr1_out,
                    fsqz1_out,
                    fsql1_out,
                    accepted,
                    r00_j,
                    z00_j,
                    w_mhd,
                    time_step_report,
                    zero_m1,
                    include_edge,
                    res0,
                    res1,
                    iter1_post,
                    bad_jacobian,
                    min_tau,
                    max_tau,
                    min_tau_ptau,
                    max_tau_ptau,
                    min_tau_state,
                    max_tau_state,
                    badjac_ptau,
                    badjac_state,
                )

            iter2_hold = jnp.asarray(it + 1, dtype=jnp.int32) + jnp.asarray(carry.iter_offset, dtype=jnp.int32)
            hold_cond = carry.converged | carry.abort_scan | (iter2_hold > jnp.asarray(int(max_iter), dtype=jnp.int32))
            return jax.lax.cond(hold_cond, _hold_step, _advance_step, operand=carry)

        carry0 = _ScanCarry(
            state=state_init,
            time_step=time_step0,
            inv_tau=inv_tau0,
            fsq_prev=fsq_prev0,
            fsq0_prev=fsq0_prev0,
            accepted_count=jnp.asarray(0, dtype=jnp.int32),
            probe_count=jnp.asarray(0, dtype=jnp.int32),
            probe_bad_jac=jnp.asarray(0, dtype=jnp.int32),
            probe_accept=jnp.asarray(0, dtype=jnp.int32),
            probe_fsq_min=jnp.asarray(jnp.inf, dtype=dtype),
            probe_fsq_max=jnp.asarray(-jnp.inf, dtype=dtype),
            probe_fsq_start=jnp.asarray(jnp.inf, dtype=dtype),
            fallback_active=jnp.asarray(True),
            abort_scan=jnp.asarray(False),
            skip_timecontrol=jnp.asarray(False),
            vRcc=vRcc0,
            vRss=vRss0,
            vZsc=vZsc0,
            vZcs=vZcs0,
            vLsc=vLsc0,
            vLcs=vLcs0,
            vRsc=vRsc0,
            vRcs=vRcs0,
            vZcc=vZcc0,
            vZss=vZss0,
            vLcc=vLcc0,
            vLss=vLss0,
            flip_sign=flip_sign0,
            iter_offset=jnp.asarray(iter_offset0, dtype=jnp.int32),
            iter1=iter1_0,
            res0=res0_0,
            res1=res1_0,
            state_checkpoint=state_checkpoint0,
            cache_valid=cache_valid0,
            cache_precond_diag=cache_precond_diag0,
            cache_tcon=cache_tcon0,
            cache_norms=cache_norms0,
            cache_rz_scale=cache_rz_scale0,
            cache_l_scale=cache_l_scale0,
            cache_rz_norm=cache_rz_norm0,
            cache_f_norm1=cache_f_norm1_0,
            cache_prec_rz_mats=cache_rz_mats0,
            cache_prec_lam_prec=cache_lam_prec0,
            force_bcovar_update=force_bcovar0,
            ijacob=ijacob0,
            bad_resets=bad_resets0,
            bad_growth=bad_growth0,
            fsqz_prev=fsqz_prev0,
            r00_prev=r00_prev0,
            z00_prev=z00_prev0,
            w_mhd_prev=w_mhd_prev0,
            converged=jnp.asarray(False),
            fsqr_prev_phys=jnp.asarray(2.0, dtype=dtype),
            fsqz_prev_phys=jnp.asarray(0.0, dtype=dtype),
            fsql_prev_phys=jnp.asarray(0.0, dtype=dtype),
            fsqr1_prev=jnp.asarray(0.0, dtype=dtype),
            fsqz1_prev=jnp.asarray(0.0, dtype=dtype),
            fsql1_prev=jnp.asarray(0.0, dtype=dtype),
            fsqr_checkpoint=jnp.asarray(0.0, dtype=dtype),
            fsqz_checkpoint=jnp.asarray(0.0, dtype=dtype),
            fsql_checkpoint=jnp.asarray(0.0, dtype=dtype),
            fsqr1_checkpoint=jnp.asarray(0.0, dtype=dtype),
            fsqz1_checkpoint=jnp.asarray(0.0, dtype=dtype),
            fsql1_checkpoint=jnp.asarray(0.0, dtype=dtype),
            edge_Rcos=jnp.asarray(edge_Rcos, dtype=dtype),
            edge_Rsin=jnp.asarray(edge_Rsin, dtype=dtype),
            edge_Zcos=jnp.asarray(edge_Zcos, dtype=dtype),
            edge_Zsin=jnp.asarray(edge_Zsin, dtype=dtype),
        )

        preflight_plan = _resolve_scan_preflight_iters(
            jit_forces_scan=bool(jit_forces_scan),
            vmec2000_control=bool(vmec2000_control),
            max_iter=int(max_iter),
            axis_reset_repeat=bool(axis_reset_repeat),
            preflight_env=os.getenv("VMEC_JAX_SCAN_PREFLIGHT"),
        )
        iteration_plan = _resolve_scan_iteration_plan(
            max_iter=int(max_iter),
            preflight_iters=int(preflight_plan.preflight_iters),
            vmec2000_control=bool(vmec2000_control),
            extra_iters_env=os.getenv("VMEC_JAX_SCAN_EXTRA_ITERS"),
        )
        preflight_iters = iteration_plan.preflight_iters
        max_iter_scan = iteration_plan.max_iter_scan
        max_iter_tail = iteration_plan.max_iter_tail

        iter_offset_preflight = iter_offset0
        if axis_reset_repeat:
            iter_offset_preflight = 0
            iter_offset0 = -1
            carry0 = carry0._replace(iter_offset=jnp.asarray(iter_offset0, dtype=jnp.int32))

        scan_cache_key = _build_vmec2000_scan_cache_key(
            static_key=static_key,
            wout_key=wout_key,
            edge_signature_key=edge_signature_key,
            tomnsps_policy_key=(
                os.getenv("VMEC_JAX_TOMNSPS_FFT", "").strip().lower(),
                os.getenv("VMEC_JAX_TOMNSPS_FFT_FUSED", "1").strip().lower(),
                os.getenv("VMEC_JAX_TOMNSPS_THETA_FUSED", "1").strip().lower(),
                os.getenv("VMEC_JAX_TOMNSPS_ZETA_FUSED", "1").strip().lower(),
            ),
            max_iter_tail=int(max_iter_tail),
            preflight_iters=int(preflight_iters),
            iter_offset0=int(iter_offset0),
            step_size=float(step_size),
            initial_flip_sign=float(initial_flip_sign),
            lambda_update_scale=float(lambda_update_scale),
            ftol=float(ftol),
            nstep_screen=int(nstep_screen),
            use_restart_triggers=bool(use_restart_triggers),
            vmecpp_restart=bool(vmecpp_restart),
            scan_use_precomputed=bool(scan_use_precomputed),
            scan_use_lax_tridi=bool(scan_use_lax_tridi),
            scan_use_restart_payload=bool(scan_use_restart_payload),
            stage_prev_fsq=stage_prev_fsq,
            stage_transition_factor=float(stage_transition_factor),
            stage_transition_scale=float(stage_transition_scale),
            jit_forces_scan=bool(jit_forces_scan),
            state_only_scan=bool(state_only_scan),
            scan_light=bool(scan_light),
            scan_minimal=bool(scan_minimal),
            scan_fallback_iters=int(scan_fallback_iters),
            scan_fallback_accept_frac=float(scan_fallback_accept_frac),
            scan_fallback_fsq_factor=float(scan_fallback_fsq_factor),
            scan_fallback_badjac_limit=int(scan_fallback_badjac_limit),
            scan_fallback_fsq_abs=float(scan_fallback_fsq_abs),
        )

        def _run_scan(carry_init, it_seq):
            return jax.lax.scan(_scan_step, carry_init, it_seq)

        def _get_scan_runner(seq_len: int):
            key = scan_cache_key + (int(seq_len),)
            if scan_differentiated:
                # A runner created during a JAX transform closes over traced
                # constants from the solve setup. Keep it local to the transform.
                if scan_timing_enabled:
                    scan_timing_stats["scan_runner_cache_bypass_count"] = (
                        int(scan_timing_stats.get("scan_runner_cache_bypass_count", 0)) + 1
                    )
                return jit(_run_scan), "bypass"
            cache_lookup_start = time.perf_counter() if scan_timing_enabled else None
            cached_run = _jit_cache_get(_SCAN_RUNNER_CACHE, key)
            if scan_timing_enabled and cache_lookup_start is not None:
                scan_timing_stats["scan_runner_cache_lookup_s"] += time.perf_counter() - float(cache_lookup_start)
            if cached_run is None:
                if scan_timing_enabled:
                    scan_timing_stats["scan_runner_cache_miss_count"] = (
                        int(scan_timing_stats.get("scan_runner_cache_miss_count", 0)) + 1
                    )
                cache_build_start = time.perf_counter() if scan_timing_enabled else None
                runner = jit(_run_scan)
                cached_runner = _jit_cache_put(
                    _SCAN_RUNNER_CACHE,
                    key,
                    runner,
                    env_name="VMEC_JAX_SCAN_RUNNER_CACHE_SIZE",
                    default=32,
                )
                if scan_timing_enabled and cache_build_start is not None:
                    scan_timing_stats["scan_runner_cache_build_s"] += time.perf_counter() - float(cache_build_start)
                return cached_runner, "miss"
            if scan_timing_enabled:
                scan_timing_stats["scan_runner_cache_hit_count"] = (
                    int(scan_timing_stats.get("scan_runner_cache_hit_count", 0)) + 1
                )
            return cached_run, "hit"

        def _emit_scan_prints(
            *,
            hist_np,
            it_start: int,
            max_iter_local: int,
        ) -> bool:
            return _emit_scan_debug_prints(
                hist_np=hist_np,
                it_start=it_start,
                max_iter_local=max_iter_local,
                scan_minimal=bool(scan_minimal),
                scan_light=bool(scan_light),
                ftol=float(ftol),
                fsq_total_target=fsq_total_target,
                iter_offset0=int(iter_offset0),
                should_print=_should_print_vmec2000_local,
                print_row=_print_vmec2000_row_local,
            )

        if scan_timing_enabled and scan_run_setup_start is not None:
            scan_timing_stats["scan_run_setup_s"] += time.perf_counter() - float(scan_run_setup_start)
        carry_init = carry0._replace(state=state_init, state_checkpoint=state_init)
        if chunked_print:
            hist_parts = []
            start_idx = 0
            carry = carry_init
            abort_scan_host = False
            fsq_min_global_j = jnp.asarray(jnp.inf, dtype=dtype)
            early_fallback = bool(scan_fallback_enabled_run) and (int(cfg.ns) > int(scan_fallback_iters))
            need_print = bool(scan_collect_print)
            chunk_size, chunk_cap_remaining = _scan_chunk_settings(
                max_iter_scan=int(max_iter_scan),
                nstep_screen=int(nstep_screen),
                need_print=bool(need_print),
                lthreed=bool(cfg.lthreed),
            )
            jit_preflight = _scan_jit_preflight_enabled(
                env_value=os.getenv("VMEC_JAX_SCAN_JIT_PREFLIGHT"),
                backend_name=_scan_backend_name(),
                scan_differentiated=bool(scan_differentiated),
            ) and (not bool(need_print))
            if preflight_iters > 0:
                t_preflight = time.perf_counter() if scan_timing_enabled else None
                if iter_offset_preflight is not None:
                    carry = carry._replace(iter_offset=jnp.asarray(iter_offset_preflight, dtype=jnp.int32))
                if jit_preflight:
                    preflight_runner, _preflight_cache_status = _get_scan_runner(1)
                    carry, hist_pre_seq = preflight_runner(carry, jnp.asarray([0], dtype=jnp.int32))
                    if scan_timing_enabled and t_preflight is not None:
                        carry, hist_pre_seq = _block_scan_value((carry, hist_pre_seq))
                    hist_pre = jax.tree_util.tree_map(lambda a: a[0], hist_pre_seq)
                else:
                    try:
                        with jax.disable_jit():
                            carry, hist_pre = _scan_step(carry, jnp.asarray(0, dtype=jnp.int32))
                    except Exception:
                        carry, hist_pre = _scan_step(carry, jnp.asarray(0, dtype=jnp.int32))
                if not state_only_scan:
                    fsq_min_global_j = jnp.minimum(
                        fsq_min_global_j,
                        jnp.min(hist_pre[0] + hist_pre[1] + hist_pre[2]),
                    )
                if need_print:
                    hist_pre_np = jax.tree_util.tree_map(lambda a: np.asarray(a)[None], hist_pre)
                    hist_parts.append(hist_pre_np)
                    _ = _emit_scan_prints(hist_np=hist_pre_np, it_start=0, max_iter_local=int(max_iter))
                elif not state_only_scan:
                    hist_parts.append(jax.tree_util.tree_map(lambda a: a[None], hist_pre))
                start_idx = int(preflight_iters)
                if axis_reset_repeat:
                    carry = carry._replace(iter_offset=jnp.asarray(iter_offset0, dtype=jnp.int32))
                if scan_timing_enabled and t_preflight is not None:
                    scan_timing_stats["scan_preflight_s"] += time.perf_counter() - float(t_preflight)
            while start_idx < int(max_iter_scan):
                # Fixed-length chunk to avoid retracing for varying tail sizes.
                # On quiet accelerator runs, cap the chunk to the remaining work
                # so short probe windows do not pay hundreds of masked no-op steps.
                remaining = int(max_iter_scan) - int(start_idx)
                if remaining <= 0:
                    break
                chunk_len = min(int(chunk_size), int(remaining)) if chunk_cap_remaining else int(chunk_size)
                it_seq = jnp.arange(start_idx, start_idx + int(chunk_len), dtype=jnp.int32)
                runner, cache_status = _get_scan_runner(int(chunk_len))
                t_device = time.perf_counter() if scan_timing_enabled else None
                carry, hist_chunk = runner(carry, it_seq)
                if scan_timing_enabled and t_device is not None:
                    carry, hist_chunk = _scan_device_run_ready(t_device, (carry, hist_chunk), cache_status=cache_status)
                if not state_only_scan:
                    fsq_min_global_j = jnp.minimum(
                        fsq_min_global_j,
                        jnp.min(hist_chunk[0] + hist_chunk[1] + hist_chunk[2]),
                    )
                if need_print:
                    hist_chunk_np = jax.tree_util.tree_map(lambda a: np.asarray(a), hist_chunk)
                    hist_parts.append(hist_chunk_np)
                    converged_now = _emit_scan_prints(
                        hist_np=hist_chunk_np,
                        it_start=int(start_idx),
                        max_iter_local=int(max_iter),
                    )
                else:
                    if not state_only_scan:
                        hist_parts.append(hist_chunk)
                    converged_now = False
                start_idx = int(start_idx + int(chunk_len))
                if (
                    scan_fallback_enabled_run
                    and int(scan_fallback_iters) > 0
                    and start_idx >= int(scan_fallback_iters)
                    and bool(np.asarray(carry.fallback_active))
                ):
                    # Disable fallback logic after the probe window completes.
                    carry = carry._replace(fallback_active=jnp.asarray(False))
                if converged_now:
                    break
                if scan_fallback_enabled_run and start_idx >= int(scan_fallback_iters):
                    # Defer host sync for fsq_min_global until after the loop.
                    pass
                if bool(np.asarray(carry.converged)) or bool(np.asarray(carry.abort_scan)):
                    break
            if scan_fallback_enabled_run and start_idx >= int(scan_fallback_iters):
                try:
                    fsq_min_global = float(jax.device_get(fsq_min_global_j))
                except Exception:
                    fsq_min_global = None
                if fsq_min_global is not None and fsq_min_global > float(scan_fallback_fsq_abs):
                    abort_scan_host = True
            if state_only_scan and not need_print:
                hist = None
            elif need_print:
                hist = jax.tree_util.tree_map(lambda *parts: np.concatenate(parts, axis=0), *hist_parts)
            else:
                t_materialize = time.perf_counter() if scan_timing_enabled else None
                hist = jax.tree_util.tree_map(lambda *parts: jnp.concatenate(parts, axis=0), *hist_parts)
                if not _tree_has_tracer(hist):
                    hist = jax.tree_util.tree_map(lambda a: np.asarray(a), hist)
                if scan_timing_enabled and t_materialize is not None:
                    scan_timing_stats["scan_host_materialize_s"] += time.perf_counter() - float(t_materialize)
            carry_final = carry
            if abort_scan_host:
                carry_final = carry_final._replace(abort_scan=jnp.asarray(True))
        else:
            runner, cache_status = _get_scan_runner(int(max_iter_tail) if int(max_iter_tail) > 0 else int(max_iter_scan))
            if preflight_iters > 0:
                # Preflight the first iteration separately to avoid XLA aliasing
                # issues in the initial tomnsps pass.  Accelerator runs may use
                # a cached one-step runner to avoid a slow host-side force pass.
                t_preflight = time.perf_counter() if scan_timing_enabled else None
                carry_pre = carry_init
                if iter_offset_preflight is not None:
                    carry_pre = carry_pre._replace(iter_offset=jnp.asarray(iter_offset_preflight, dtype=jnp.int32))
                jit_preflight = _scan_jit_preflight_enabled(
                    env_value=os.getenv("VMEC_JAX_SCAN_JIT_PREFLIGHT"),
                    backend_name=_scan_backend_name(),
                    scan_differentiated=bool(scan_differentiated),
                ) and (not bool(scan_collect_print))
                if jit_preflight:
                    preflight_runner, _preflight_cache_status = _get_scan_runner(1)
                    carry_pre, hist_pre_seq = preflight_runner(carry_pre, jnp.asarray([0], dtype=jnp.int32))
                    hist_pre = jax.tree_util.tree_map(lambda a: a[0], hist_pre_seq)
                else:
                    try:
                        with jax.disable_jit():
                            carry_pre, hist_pre = _scan_step(carry_pre, jnp.asarray(0, dtype=jnp.int32))
                    except Exception:
                        carry_pre, hist_pre = _scan_step(carry_pre, jnp.asarray(0, dtype=jnp.int32))
                if (
                    scan_fallback_enabled_run
                    and int(scan_fallback_iters) > 0
                    and int(preflight_iters) >= int(scan_fallback_iters)
                ):
                    carry_pre = carry_pre._replace(fallback_active=jnp.asarray(False))
                if scan_timing_enabled and t_preflight is not None:
                    carry_pre, hist_pre = _block_scan_value((carry_pre, hist_pre))
                    scan_timing_stats["scan_preflight_s"] += time.perf_counter() - float(t_preflight)
                if max_iter_tail > 0:
                    it_seq = jnp.arange(preflight_iters, int(max_iter_scan), dtype=jnp.int32)
                    if axis_reset_repeat:
                        carry_pre = carry_pre._replace(iter_offset=jnp.asarray(iter_offset0, dtype=jnp.int32))
                    t_device = time.perf_counter() if scan_timing_enabled else None
                    carry_final, hist_tail = runner(carry_pre, it_seq)
                    if scan_timing_enabled and t_device is not None:
                        carry_final, hist_tail = _scan_device_run_ready(
                            t_device,
                            (carry_final, hist_tail),
                            cache_status=cache_status,
                        )
                    if state_only_scan:
                        hist = None
                    else:
                        hist = jax.tree_util.tree_map(
                            lambda a, b: jnp.concatenate([a[None], b], axis=0),
                            hist_pre,
                            hist_tail,
                        )
                else:
                    carry_final = carry_pre
                    hist = None if state_only_scan else jax.tree_util.tree_map(lambda a: a[None], hist_pre)
            else:
                it_seq = jnp.arange(int(max_iter_scan), dtype=jnp.int32)
                t_device = time.perf_counter() if scan_timing_enabled else None
                carry_final, hist = runner(carry_init, it_seq)
                if scan_timing_enabled and t_device is not None:
                    carry_final, hist = _scan_device_run_ready(t_device, (carry_final, hist), cache_status=cache_status)
                if state_only_scan:
                    hist = None
        scan_postprocess_start = time.perf_counter() if scan_timing_enabled else None
        if state_only_scan:
            traced = _tree_has_tracer(carry_final.state)
            hist_dtype = jnp.asarray(state0.Rcos).dtype
            empty = jnp.zeros((0,), dtype=hist_dtype) if traced else np.asarray([], dtype=float)
            scan_timing_report = None
            if scan_timing_enabled:
                if scan_postprocess_start is not None:
                    scan_timing_stats["scan_postprocess_s"] += time.perf_counter() - float(scan_postprocess_start)
                scan_total_s = (
                    time.perf_counter() - float(scan_total_start)
                    if scan_total_start is not None
                    else sum(scan_timing_stats.values())
                )
                scan_timing_report = _build_scan_timing_report(
                    iterations=int(max_iter),
                    stats=scan_timing_stats,
                    scan_total_s=float(scan_total_s),
                )
            diagnostics = {
                "use_scan": True,
                "vmec2000_scan": True,
                "scan_path": "vmec2000",
                "state_only": True,
                "history_mode": "none",
                "history_none": True,
                "ftol": float(ftol),
                "requested_ftol": float(ftol),
                "scan_minimal": bool(scan_minimal),
                "light_history": bool(scan_light),
                "scan_use_precomputed": bool(scan_use_precomputed),
                "scan_use_lax_tridi": bool(scan_use_lax_tridi),
                **({"timing": scan_timing_report} if scan_timing_report is not None else {}),
            }
            if not traced:
                diagnostics.update(
                    {
                        "abort_scan": bool(np.asarray(carry_final.abort_scan)),
                        "converged": bool(np.asarray(carry_final.converged)),
                        "ijacob": int(np.asarray(carry_final.ijacob)),
                    }
                )
            return _attach_freeb_diag(
                SolveVmecResidualResult(
                    state=carry_final.state,
                    n_iter=int(max_iter),
                    w_history=empty,
                    fsqr2_history=empty,
                    fsqz2_history=empty,
                    fsql2_history=empty,
                    grad_rms_history=empty,
                    step_history=empty,
                    diagnostics=diagnostics,
                )
            )
        scan_histories = unpack_vmec2000_scan_histories(
            hist,
            scan_minimal=bool(scan_minimal),
            scan_light=bool(scan_light),
        )
        if _tree_has_tracer(hist) or _tree_has_tracer(carry_final.state):
            hist_dtype = jnp.asarray(state0.Rcos).dtype
            empty = jnp.zeros((0,), dtype=hist_dtype)
            traced_resume_state = {
                "time_step": carry_final.time_step,
                "inv_tau": carry_final.inv_tau,
                "fsq_prev": carry_final.fsq_prev,
                "fsq0_prev": carry_final.fsq0_prev,
                "flip_sign": carry_final.flip_sign,
                "iter1": carry_final.iter1,
                "iter_offset": carry_final.iter_offset + jnp.asarray(int(max_iter), dtype=jnp.int32),
                "res0": carry_final.res0,
                "res1": carry_final.res1,
                "ijacob": carry_final.ijacob,
                "bad_resets": carry_final.bad_resets,
                "bad_growth_streak": carry_final.bad_growth,
                "fsqz_prev": carry_final.fsqz_prev,
                "state_checkpoint": carry_final.state_checkpoint,
                "vRcc": carry_final.vRcc,
                "vRss": carry_final.vRss,
                "vZsc": carry_final.vZsc,
                "vZcs": carry_final.vZcs,
                "vLsc": carry_final.vLsc,
                "vLcs": carry_final.vLcs,
                "vRsc": carry_final.vRsc,
                "vRcs": carry_final.vRcs,
                "vZcc": carry_final.vZcc,
                "vZss": carry_final.vZss,
                "vLcc": carry_final.vLcc,
                "vLss": carry_final.vLss,
                "vmec2000_cache_valid": carry_final.cache_valid,
                "force_bcovar_update": carry_final.force_bcovar_update,
            }
            return _attach_freeb_diag(
                SolveVmecResidualResult(
                    state=carry_final.state,
                    n_iter=int(max_iter),
                    w_history=empty,
                    fsqr2_history=empty,
                    fsqz2_history=empty,
                    fsql2_history=empty,
                    grad_rms_history=empty,
                    step_history=empty,
                    diagnostics={
                        "use_scan": True,
                        "vmec2000_scan": True,
                        "scan_path": "vmec2000",
                        "traced_scan": True,
                        "scan_use_precomputed": bool(scan_use_precomputed),
                        "scan_use_lax_tridi": bool(scan_use_lax_tridi),
                        "resume_state": traced_resume_state,
                    },
                )
            )
        scan_output = postprocess_vmec2000_scan_result(
            scan_histories,
            carry_final,
            vmec2000_control=bool(vmec2000_control),
            ftol=float(ftol),
            fsq_total_target=fsq_total_target,
            max_iter=int(max_iter),
            scan_minimal=bool(scan_minimal),
            scan_light=bool(scan_light),
            resume_state_mode=str(resume_state_mode),
            pack_resume_state=_pack_resume_state,
            free_boundary_enabled=bool(free_boundary_enabled),
            freeb_nvacskip=int(freeb_nvacskip),
            freeb_nvskip0=int(freeb_nvskip0),
            iter_offset0=int(iter_offset0),
            free_boundary_iter_controls=_free_boundary_iter_controls,
        )
        fsqr_full = scan_output.fsqr_full
        fsqz_full = scan_output.fsqz_full
        fsql_full = scan_output.fsql_full
        conv_idx_print = scan_output.conv_idx_print

        if (
            (not scan_minimal)
            and (not print_in_scan)
            and (not chunked_print)
            and verbose
            and bool(vmec2000_control)
            and bool(verbose_vmec2000_table)
        ):
            r00_full = np.asarray(scan_histories.r00)
            z00_full = np.asarray(scan_histories.z00)
            w_mhd_full = np.asarray(scan_histories.w_mhd)
            dt_full = np.asarray(scan_histories.dt)
            last_iter = int(conv_idx_print) if int(conv_idx_print) > 0 else int(max_iter)
            for i in range(last_iter):
                iter2 = i + 1
                if _should_print_vmec2000_local(int(iter2), int(last_iter)):
                    r00_val = float(r00_full[i])
                    z00_val = float(z00_full[i])
                    # Match VMEC precision (E11.3) for r00/z00.
                    r00_val = float(f"{r00_val:.3E}")
                    z00_val = float(f"{z00_val:.3E}")
                    _print_vmec2000_row_local(
                        iter_idx=int(iter2),
                        fsqr=float(fsqr_full[i]),
                        fsqz=float(fsqz_full[i]),
                        fsql=float(fsql_full[i]),
                        delt0r=float(dt_full[i]),
                        r00=r00_val,
                        w_mhd=float(w_mhd_full[i]),
                        z00=z00_val,
                    )
        if (not scan_light) and (not scan_minimal) and os.getenv("VMEC_JAX_DUMP_PTAU", "") not in ("", "0"):
            last_iter = int(conv_idx_print) if int(conv_idx_print) > 0 else int(max_iter)
            ptau_min_full = np.asarray(scan_histories.ptau_min)
            ptau_max_full = np.asarray(scan_histories.ptau_max)
            tau_min_state_full = np.asarray(scan_histories.tau_min_state)
            tau_max_state_full = np.asarray(scan_histories.tau_max_state)
            badjac_ptau_full = np.asarray(scan_histories.badjac_ptau).astype(int)
            badjac_state_full = np.asarray(scan_histories.badjac_state).astype(int)
            for i in range(last_iter):
                iter2 = i + 1 + int(iter_offset0)
                _maybe_dump_ptau(
                    iter_idx=int(iter2),
                    ptau_min=float(ptau_min_full[i]),
                    ptau_max=float(ptau_max_full[i]),
                    tau_min_state=float(tau_min_state_full[i]) if np.isfinite(tau_min_state_full[i]) else None,
                    tau_max_state=float(tau_max_state_full[i]) if np.isfinite(tau_max_state_full[i]) else None,
                    badjac_ptau=bool(badjac_ptau_full[i]),
                    badjac_state=bool(badjac_state_full[i]),
                    badjac_used=bool(np.asarray(scan_histories.bad_jac)[i]),
                    mode=badjac_mode,
                    label="scan",
                )
        n_iter_hist = scan_output.n_iter_hist
        scan_timing_report = None
        if scan_timing_enabled:
            if scan_postprocess_start is not None:
                scan_timing_stats["scan_postprocess_s"] += time.perf_counter() - float(scan_postprocess_start)
            scan_total_s = (
                time.perf_counter() - float(scan_total_start)
                if scan_total_start is not None
                else sum(scan_timing_stats.values())
            )
            scan_timing_report = _build_scan_timing_report(
                iterations=int(n_iter_hist),
                stats=scan_timing_stats,
                scan_total_s=float(scan_total_s),
            )
        res_scan = SolveVmecResidualResult(
            state=carry_final.state,
            n_iter=int(scan_output.w_history.shape[0]),
            w_history=np.asarray(scan_output.w_history),
            fsqr2_history=np.asarray(scan_output.fsqr_history),
            fsqz2_history=np.asarray(scan_output.fsqz_history),
            fsql2_history=np.asarray(scan_output.fsql_history),
            grad_rms_history=np.asarray([], dtype=float),
            step_history=np.asarray([], dtype=float),
            diagnostics={
                "use_scan": True,
                "vmec2000_scan": True,
                "scan_path": "vmec2000",
                "ftol": float(ftol),
                "requested_ftol": float(ftol),
                "light_history": bool(scan_light),
                "scan_minimal": bool(scan_minimal),
                "scan_use_precomputed": bool(scan_use_precomputed),
                "scan_use_lax_tridi": bool(scan_use_lax_tridi),
                "resume_state_mode": str(resume_state_mode),
                "fsq_total_target": fsq_total_target,
                "badjac_use_state": bool(badjac_use_state),
                "badjac_mode": badjac_mode,
                "badjac_state_probe": bool(badjac_state_probe),
                "badjac_initial_state_probe_iters": int(badjac_initial_state_probe_iters),
                "ijacob": int(np.asarray(carry_final.ijacob)),
                "abort_scan": bool(np.asarray(carry_final.abort_scan)),
                **scan_output.diagnostics,
                **({"timing": scan_timing_report} if scan_timing_report is not None else {}),
            },
        )
        return _attach_freeb_diag(res_scan)

    if use_scan:
        if vmec2000_control:
            scan_result = _run_vmec2000_scan(state)
            if scan_fallback_enabled and (not bool(state_only)):
                fallback_decision = _scan_fallback_decision(
                    diagnostics=scan_result.diagnostics,
                    fsqr_history=scan_result.fsqr2_history,
                    fsqz_history=scan_result.fsqz2_history,
                    fsql_history=scan_result.fsql2_history,
                    max_iter=int(max_iter),
                    fallback_iters=int(scan_fallback_iters),
                    badjac_limit=int(scan_fallback_badjac_limit),
                    fsq_abs=float(scan_fallback_fsq_abs),
                    accept_frac=float(scan_fallback_accept_frac),
                    fsq_factor=float(scan_fallback_fsq_factor),
                )
                if fallback_decision.fallback:
                    if verbose:
                        print(
                            "[solve_fixed_boundary_residual_iter] "
                            f"scan fallback -> non-scan ({fallback_decision.reason_text})"
                            f"{fallback_decision.probe_message}",
                            flush=True,
                        )
                    use_scan = False
                    resume_state = None
                    state = state0
                else:
                    return _attach_freeb_diag(scan_result)
            else:
                return _attach_freeb_diag(scan_result)

        if use_scan:
            if (
                backtracking
                or use_restart_triggers
                or auto_flip_force
                or limit_dt_from_force
                or limit_update_rms
                or strict_update
                or use_direct_fallback
                or reference_mode
            ):
                raise ValueError(
                    "use_scan requires vmec2000_control=False, backtracking=False, "
                    "use_restart_triggers=False, auto_flip_force=False, "
                    "limit_dt_from_force=False, limit_update_rms=False, strict_update=False, "
                    "use_direct_fallback=False, reference_mode=False."
                )

            scan_timing_enabled = _scan_timing_enabled(os.getenv("VMEC_JAX_TIMING", ""))
            scan_timing_stats = _new_scan_timing_stats()
            scan_total_start = time.perf_counter() if scan_timing_enabled else None

            dtype = jnp.asarray(state0.Rcos).dtype
            time_step_j = jnp.asarray(float(step_size), dtype=dtype)
            flip_sign_j = jnp.asarray(float(initial_flip_sign), dtype=dtype)
            ftol_j = jnp.asarray(float(ftol), dtype=dtype)
            fsq_total_target_j = None
            if fsq_total_target is not None:
                fsq_total_target_j = jnp.asarray(float(fsq_total_target), dtype=dtype)

            def _converged_residuals_scan_fast(fsqr, fsqz, fsql):
                return _runtime_converged_residuals_scan_fast(
                    fsqr,
                    fsqz,
                    fsql,
                    ftol=ftol_j,
                    fsq_total_target=fsq_total_target_j,
                )

            include_edge_scan = False
            _compute_forces_scan = _compute_forces if jit_forces else _compute_forces_impl

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
                scan_timing_stats["scan_setup_s"] += time.perf_counter() - float(scan_total_start)

            def _scan_step(carry, it):
                state, converged, converged_iter, last_fsqr, last_fsqz, last_fsql = carry
                it = jnp.asarray(it, dtype=jnp.int32)

                def _hold_step(_):
                    return carry, (last_fsqr, last_fsqz, last_fsql)

                def _advance_step(_):
                    iter_since_restart = it + 1
                    zero_m1 = jnp.where(
                        iter_since_restart < 2, jnp.asarray(1.0, dtype=dtype), jnp.asarray(0.0, dtype=dtype)
                    )

                    _k, frzl, fsqr, fsqz, fsql, rz_scale, l_scale, _norms = _compute_forces_scan(
                        state,
                        include_edge=include_edge_scan,
                        zero_m1=zero_m1,
                        iter_idx=None,
                    )
                    frss_in = (frzl.frss if frzl.frss is not None else jnp.zeros_like(frzl.frcc)) * rz_scale[
                        :, None, None
                    ]
                    fzcs_in = (frzl.fzcs if frzl.fzcs is not None else jnp.zeros_like(frzl.fzsc)) * rz_scale[
                        :, None, None
                    ]
                    frcc, frss, fzsc, fzcs = _apply_radial_tridi_batched(
                        [
                            frzl.frcc * rz_scale[:, None, None],
                            frss_in,
                            frzl.fzsc * rz_scale[:, None, None],
                            fzcs_in,
                        ],
                        precond_radial_alpha,
                    )
                    flcs_in = (frzl.flcs if frzl.flcs is not None else jnp.zeros_like(frzl.flsc)) * l_scale[
                        :, None, None
                    ]
                    flsc, flcs = _apply_radial_tridi_batched(
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
                        jnp.asarray(getattr(frzl, "frsc", None))
                        if getattr(frzl, "frsc", None) is not None
                        else jnp.zeros_like(frcc_u)
                    ) * w_mode_mn[None, :, :]
                    frcs_u = (
                        jnp.asarray(getattr(frzl, "frcs", None))
                        if getattr(frzl, "frcs", None) is not None
                        else jnp.zeros_like(frcc_u)
                    ) * w_mode_mn[None, :, :]
                    fzcc_u = (
                        jnp.asarray(getattr(frzl, "fzcc", None))
                        if getattr(frzl, "fzcc", None) is not None
                        else jnp.zeros_like(fzsc_u)
                    ) * w_mode_mn[None, :, :]
                    fzss_u = (
                        jnp.asarray(getattr(frzl, "fzss", None))
                        if getattr(frzl, "fzss", None) is not None
                        else jnp.zeros_like(fzsc_u)
                    ) * w_mode_mn[None, :, :]
                    flcc_u = (
                        jnp.asarray(getattr(frzl, "flcc", None))
                        if getattr(frzl, "flcc", None) is not None
                        else jnp.zeros_like(flsc_u)
                    ) * w_mode_mn[None, :, :]
                    flss_u = (
                        jnp.asarray(getattr(frzl, "flss", None))
                        if getattr(frzl, "flss", None) is not None
                        else jnp.zeros_like(flsc_u)
                    ) * w_mode_mn[None, :, :]

                    if lambda_update_scale != 1.0:
                        flsc_u = flsc_u * lambda_update_scale_j
                        flcs_u = flcs_u * lambda_update_scale_j
                        flcc_u = flcc_u * lambda_update_scale_j
                        flss_u = flss_u * lambda_update_scale_j

                    dR = (time_step_j * flip_sign_j) * _mn_cos_to_signed_physical(frcc_u, frss_u)
                    sin_updates = _mn_sin_to_signed_batch(
                        jnp.stack([fzsc_u, flsc_u], axis=0),
                        jnp.stack([fzcs_u, flcs_u], axis=0),
                    )
                    dZ = (time_step_j * flip_sign_j) * sin_updates[0]
                    dL = (time_step_j * flip_sign_j) * sin_updates[1]
                    if bool(cfg.lasym):
                        dR_sin = (time_step_j * flip_sign_j) * _mn_sin_to_signed_physical(frsc_u, frcs_u)
                        dZ_cos = (time_step_j * flip_sign_j) * _mn_cos_to_signed_physical(fzcc_u, fzss_u)
                        dL_cos = (time_step_j * flip_sign_j) * _mn_cos_to_signed_physical_lambda(flcc_u, flss_u)
                    else:
                        dR_sin = jnp.zeros_like(dR)
                        dZ_cos = jnp.zeros_like(dR)
                        dL_cos = jnp.zeros_like(dR)

                    state_new = VMECState(
                        layout=state.layout,
                        Rcos=jnp.asarray(state.Rcos) + dR,
                        Rsin=jnp.asarray(state.Rsin) + dR_sin,
                        Zcos=jnp.asarray(state.Zcos) + dZ_cos,
                        Zsin=jnp.asarray(state.Zsin) + dZ,
                        Lcos=jnp.asarray(state.Lcos) + dL_cos,
                        Lsin=jnp.asarray(state.Lsin) + dL,
                    )
                    state_new = _enforce_fixed_boundary_and_axis(
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
                    state_new = _apply_vmec_lambda_axis_rules(state_new)
                    conv_now = _converged_residuals_scan_fast(fsqr, fsqz, fsql)
                    conv_iter_new = jnp.where(
                        (converged_iter < 0) & conv_now,
                        it + jnp.asarray(1, dtype=jnp.int32),
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

                return jax.lax.cond(converged, _hold_step, _advance_step, operand=None)

            def _run_scan(state_init):
                carry0 = (
                    state_init,
                    jnp.asarray(False),
                    jnp.asarray(-1, dtype=jnp.int32),
                    jnp.asarray(jnp.inf, dtype=dtype),
                    jnp.asarray(jnp.inf, dtype=dtype),
                    jnp.asarray(jnp.inf, dtype=dtype),
                )
                return jax.lax.scan(_scan_step, carry0, jnp.arange(max_iter, dtype=jnp.int32))

            scan_run_setup_start = time.perf_counter() if scan_timing_enabled else None
            cached_run = None if differentiating_scan else _jit_cache_get(_SCAN_RUNNER_CACHE, scan_cache_key)
            scan_runner_cache_status = "bypass" if differentiating_scan else ("miss" if cached_run is None else "hit")
            if scan_timing_enabled:
                if scan_run_setup_start is not None:
                    scan_timing_stats["scan_runner_cache_lookup_s"] += time.perf_counter() - float(
                        scan_run_setup_start
                    )
                if differentiating_scan:
                    scan_timing_stats["scan_runner_cache_bypass_count"] = (
                        int(scan_timing_stats.get("scan_runner_cache_bypass_count", 0)) + 1
                    )
            if cached_run is None:
                if scan_timing_enabled and (not differentiating_scan):
                    scan_timing_stats["scan_runner_cache_miss_count"] = (
                        int(scan_timing_stats.get("scan_runner_cache_miss_count", 0)) + 1
                    )
                cache_build_start = time.perf_counter() if scan_timing_enabled else None
                _run_scan = jit(_run_scan)
                if not differentiating_scan:
                    _run_scan = _jit_cache_put(
                        _SCAN_RUNNER_CACHE,
                        scan_cache_key,
                        _run_scan,
                        env_name="VMEC_JAX_SCAN_RUNNER_CACHE_SIZE",
                        default=32,
                    )
                if scan_timing_enabled and cache_build_start is not None:
                    scan_timing_stats["scan_runner_cache_build_s"] += time.perf_counter() - float(cache_build_start)
            else:
                if scan_timing_enabled:
                    scan_timing_stats["scan_runner_cache_hit_count"] = (
                        int(scan_timing_stats.get("scan_runner_cache_hit_count", 0)) + 1
                    )
                _run_scan = cached_run
            if scan_timing_enabled and scan_run_setup_start is not None:
                scan_timing_stats["scan_run_setup_s"] += time.perf_counter() - float(scan_run_setup_start)

            scan_device_start = time.perf_counter() if scan_timing_enabled else None
            carry_final, hist = _run_scan(state)
            if scan_timing_enabled and scan_device_start is not None:
                carry_final, hist = _runtime_scan_device_run_ready(
                    start=scan_device_start,
                    value=(carry_final, hist),
                    scan_timing_enabled=True,
                    perf_counter=time.perf_counter,
                    block_until_ready=jax.block_until_ready,
                    tree_map=jax.tree_util.tree_map,
                    record_ready=_record_scan_device_ready,
                    stats=scan_timing_stats,
                    cache_status=scan_runner_cache_status,
                )
            scan_materialize_start = time.perf_counter() if scan_timing_enabled else None
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
                scan_timing_stats["scan_host_materialize_s"] += time.perf_counter() - float(scan_materialize_start)
            scan_timing_report = None
            if scan_timing_enabled:
                scan_total_s = (
                    time.perf_counter() - float(scan_total_start)
                    if scan_total_start is not None
                    else sum(scan_timing_stats.values())
                )
                scan_timing_report = _build_scan_timing_report(
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
            return _attach_freeb_diag(res_scan_fast)

    profile_window = os.getenv("VMEC_JAX_PROFILE_WINDOW", "").strip().lower()
    profile_dir_env = os.getenv("VMEC_JAX_PROFILE_DIR", "").strip()
    profile_started = False
    profile_active = False
    profile_start_iter = None
    profile_dir = ""
    if profile_window and profile_dir_env:
        if profile_window in ("pre", "iter1", "1"):
            profile_start_iter = 1
        else:
            window_str = profile_window
            if window_str.startswith("iter"):
                window_str = window_str[4:]
            try:
                profile_start_iter = max(1, int(window_str))
            except Exception:
                profile_start_iter = None
        if profile_start_iter is not None:
            profile_dir = str(Path(profile_dir_env) / f"window_{profile_window}")
            profile_active = True
    perfetto_env = os.getenv("VMEC_JAX_PROFILE_PERFETTO", "1")
    profile_perfetto = perfetto_env.strip().lower() not in ("", "0", "false", "no")

    timing_stats = {
        "setup_total": 0.0,
        "setup_static_grid_rebuild": float(_setup_phase_timings["setup_static_grid_rebuild"]),
        "setup_freeb_policy": float(_setup_phase_timings["setup_freeb_policy"]),
        "setup_boundary_profiles": float(_setup_phase_timings["setup_boundary_profiles"]),
        "setup_cache_key_hash": float(_setup_phase_timings["setup_cache_key_hash"]),
        "setup_ptau_constants": float(_setup_phase_timings["setup_ptau_constants"]),
        "setup_index_constants": float(_setup_phase_timings["setup_index_constants"]),
        "setup_update_constants": float(_setup_phase_timings["setup_update_constants"]),
        "setup_axis_reset": 0.0,
        "setup_axis_reset_compute_forces": 0.0,
        "iteration_loop": 0.0,
        "iteration_prepare": 0.0,
        "iteration_residual_metrics": 0.0,
        "iteration_control": 0.0,
        "iteration_control_fsq1": 0.0,
        "iteration_control_badjac": 0.0,
        "iteration_control_vmec_time": 0.0,
        "iteration_control_restart": 0.0,
        "iteration_control_evolve": 0.0,
        "iteration_post_update": 0.0,
        "iteration_loop_unattributed": 0.0,
        "finalize": 0.0,
        "compute_forces": 0.0,
        "compute_forces_first": 0.0,
        "compute_forces_rest": 0.0,
        "compute_forces_calls": 0,
        "compute_forces_main": 0.0,
        "compute_forces_main_calls": 0,
        "compute_forces_auto_flip": 0.0,
        "compute_forces_auto_flip_calls": 0,
        "compute_forces_trial": 0.0,
        "compute_forces_trial_calls": 0,
        "compute_forces_backtracking": 0.0,
        "compute_forces_backtracking_calls": 0,
        "preconditioner": 0.0,
        "precond_apply": 0.0,
        "precond_mode_scale": 0.0,
        "update": 0.0,
        "update_state": 0.0,
        "update_trace_build": 0.0,
        "update_trace_finalize": 0.0,
        "precond_refresh": 0.0,
        "iterations": 0,
    }

    def _record_compute_force_timing(label: str, start: float | None, ready_value: Any) -> None:
        if not bool(timing_enabled) or start is None:
            return
        try:
            if has_jax():
                jax.block_until_ready(ready_value)
        except Exception:
            pass
        compute_dt = time.perf_counter() - float(start)
        if label == "main":
            timing_stats["compute_forces"] += compute_dt
            if int(timing_stats["compute_forces_calls"]) == 0:
                timing_stats["compute_forces_first"] += compute_dt
            else:
                timing_stats["compute_forces_rest"] += compute_dt
            timing_stats["compute_forces_calls"] = int(timing_stats["compute_forces_calls"]) + 1
        key = f"compute_forces_{label}"
        calls_key = f"{key}_calls"
        if key in timing_stats:
            timing_stats[key] += compute_dt
        if calls_key in timing_stats:
            timing_stats[calls_key] = int(timing_stats[calls_key]) + 1

    w_history = []
    fsqr2_history = []
    fsqz2_history = []
    fsql2_history = []
    r00_history: list[float] = []
    z00_history: list[float] = []
    wb_history: list[float] = []
    wp_history: list[float] = []
    w_vmec_history: list[float] = []
    fsqr1_history = []
    fsqz1_history = []
    fsql1_history = []
    fsq1_history = []
    rz_norm_history: list[float] = []
    f_norm1_history: list[float] = []
    gcr2_p_history: list[float] = []
    gcz2_p_history: list[float] = []
    gcl2_p_history: list[float] = []
    step_status_history: list[str] = []
    restart_reason_history: list[str] = []
    pre_restart_reason_history: list[str] = []
    time_step_history: list[float] = []
    res0_history: list[float] = []
    res1_history: list[float] = []
    fsq_prev_history: list[float] = []
    bad_growth_streak_history: list[int] = []
    iter1_history: list[int] = []
    iter2_history: list[int] = []
    include_edge_history: list[int] = []
    zero_m1_history: list[int] = []
    freeb_ivac_history: list[int] = []
    freeb_ivacskip_history: list[int] = []
    freeb_full_update_history: list[int] = []
    freeb_nestor_reused_history: list[int] = []
    freeb_nestor_source_reused_history: list[int] = []
    freeb_nestor_provider_allows_source_reuse_history: list[int] = []
    freeb_nestor_bnormal_rms_history: list[float] = []
    freeb_nestor_gsource_rms_history: list[float] = []
    freeb_nestor_bsqvac_rms_history: list[float] = []
    freeb_nestor_solve_time_history: list[float] = []
    freeb_nestor_sample_time_history: list[float] = []
    freeb_nestor_trial_reused_history: list[int] = []
    freeb_nestor_trial_solve_time_history: list[float] = []
    freeb_nestor_trial_sample_time_history: list[float] = []
    freeb_nestor_trial_failed_history: list[int] = []
    dt_eff_history: list[float] = []
    update_rms_history: list[float] = []
    w_curr_history: list[float] = []
    w_try_history: list[float] = []
    w_try_ratio_history: list[float] = []
    restart_path_history: list[str] = []
    adjoint_step_trace_history: list[dict[str, Any]] = []
    min_tau_history: list[float] = []
    max_tau_history: list[float] = []
    bad_jacobian_history: list[int] = []
    grad_rms_history = []
    step_history = []
    r00_last = float("nan")
    z00_last = float("nan")
    wb_last = float("nan")
    wp_last = float("nan")
    w_vmec_last = float("nan")

    # Conjugate-gradient-like time-stepping state.
    time_step = float(step_size)
    k_ndamp = 10
    inv_tau = [0.15 / time_step] * k_ndamp
    fsq_prev = 1.0
    fsq0_prev = 1.0
    velocity_shape = (int(state.Rcos.shape[0]), mpol, nrange)
    if bool(host_update_assembly) and (not _tree_has_tracer(state.Rcos)):
        vRcc = np.zeros(velocity_shape, dtype=np.asarray(state.Rcos).dtype)
    else:
        vRcc = jnp.zeros(velocity_shape, dtype=jnp.asarray(state.Rcos).dtype)
    (
        vRss,
        vZsc,
        vZcs,
        vLsc,
        vLcs,
        vRsc,
        vRcs,
        vZcc,
        vZss,
        vLcc,
        vLss,
    ) = _zero_velocity_blocks_like(vRcc, vRcc, vRcc, vRcc, vRcc, vRcc, vRcc, vRcc, vRcc, vRcc, vRcc)
    flip_sign = float(initial_flip_sign)
    max_coeff_delta_rms = 1e-5
    max_update_rms = 5e-3
    if bool(reference_mode):
        max_coeff_delta_rms = 5e-6
        max_update_rms = 1e-3
    ijacob = 0
    bad_resets = 0
    iter1 = 1
    # VMEC runvmec/funct3d cadence starts free-boundary control at ivac=0.
    # Starting at -1 delays vacuum turn-on by one accepted iteration.
    # VMEC initializes ivac=-1 (reset_params.f), then promotes to 0/1/...
    # once free-boundary activation criteria are met.
    freeb_ivac = -1
    freeb_ivacskip = 0
    freeb_nestor_runtime: NestorRuntimeState | None = None
    freeb_bsqvac_half_current = None
    freeb_last_model = "none"
    freeb_last_diagnostics: dict[str, Any] = {}
    freeb_plascur = 0.0
    try:
        icurv_arr = np.asarray(getattr(wout_like, "icurv", np.asarray([0.0], dtype=float)), dtype=float)
        if icurv_arr.size > 0:
            freeb_plascur = float((2.0 * np.pi) * icurv_arr[-1])
    except Exception:
        freeb_plascur = 0.0

    def _vmec_freeb_plascur_from_bcovar(bc_obj, fallback: float) -> float:
        """VMEC `ctor` proxy used by NESTOR (`vacuum_par(..., ctor, ...)`)."""
        try:
            from .vmec_lforbal import plascur_edge_from_bcovar
            return _runtime_vmec_freeb_plascur_from_bcovar(
                bc_obj,
                fallback=fallback,
                plascur_edge_from_bcovar=plascur_edge_from_bcovar,
                trig=trig,
                wout=wout_like,
                s=s,
            )
        except Exception:
            return float(fallback)
        return float(fallback)

    res0 = -1.0
    k_preconditioner_update_interval = 25
    state_checkpoint = state
    bad_growth_streak = 0
    # Restart trigger factors:
    # - bad_jacobian: time_step *= 0.9
    # - bad_progress: time_step /= 1.03
    restart_badjac_factor = 0.9
    restart_badprog_factor = 1.03
    huge_force_restart_count = 0
    huge_force_restart_budget = 2
    res1 = -1.0
    vmec2000_fact = 1.0e4

    # Edge-force gating uses the *previous* iteration's residual (the first
    # iteration initializes forces to 1.0). Track that explicitly.
    prev_rz_fsq = 2.0

    debug_print_config = _resolve_debug_print_config(
        print_env=os.getenv("VMEC_JAX_SCAN_PRINT", "1"),
        mode_env=os.getenv("VMEC_JAX_SCAN_PRINT_MODE", "debug_print"),
        ordered_env=os.getenv("VMEC_JAX_SCAN_PRINT_ORDERED", "0"),
    )
    scan_print_mode = debug_print_config.mode
    scan_print_ordered = debug_print_config.ordered
    print_live = debug_print_config.print_live
    _jax_debug = None
    _io_callback = None
    if print_live:
        try:
            from jax import debug as _jax_debug  # type: ignore[assignment]
        except Exception:
            _jax_debug = None
    if scan_print_mode == "io_callback":
        try:
            from jax.experimental import io_callback as _io_callback  # type: ignore[assignment]
        except Exception:
            scan_print_mode = _resolve_debug_print_config(
                print_env="1",
                mode_env=scan_print_mode,
                ordered_env="0",
                io_callback_available=False,
            ).mode
            _io_callback = None

    def _print_vmec2000_iter_row(
        *,
        iter_idx: int,
        fsqr: float,
        fsqz: float,
        fsql: float,
        fsqr1: float,
        fsqz1: float,
        fsql1: float,
        delt0r: float,
        r00: float,
        w_mhd: float,
        z00: float | None = None,
    ) -> None:
        _emit_scan_vmec2000_iter_row(
            iter_idx=iter_idx,
            fsqr=fsqr,
            fsqz=fsqz,
            fsql=fsql,
            delt0r=delt0r,
            r00=r00,
            w_mhd=w_mhd,
            lasym=bool(cfg.lasym),
            z00=z00,
            verbose=bool(verbose),
            vmec2000_control=bool(vmec2000_control),
            verbose_vmec2000_table=bool(verbose_vmec2000_table),
            print_live=bool(print_live),
            scan_print_mode=scan_print_mode,
            scan_print_ordered=bool(scan_print_ordered),
            jax_debug=_jax_debug,
            io_callback=_io_callback,
            print_row=_print_scan_vmec2000_row,
        )

    nstep_screen = _resolve_nstep_screen(
        indata_nstep=int(indata.get_int("NSTEP", 1)) if indata is not None else 1,
        override_env=os.getenv("VMEC_JAX_NSTEP_OVERRIDE", ""),
    )

    def _should_print_vmec2000(iter_idx: int, max_iter: int) -> bool:
        return _should_print_vmec2000_row(
            iter_idx=iter_idx,
            max_iter=max_iter,
            nstep_screen=nstep_screen,
            verbose=bool(verbose),
            vmec2000_control=bool(vmec2000_control),
            verbose_vmec2000_table=bool(verbose_vmec2000_table),
        )

    def _should_sample_vmec2000(iter_idx: int, max_iter: int) -> bool:
        """Sample VMEC2000 scalar diagnostics on the screen cadence."""
        return _vmec2000_cadence_selected(
            iter_idx=iter_idx,
            max_iter=max_iter,
            nstep_screen=nstep_screen,
        )

    # VMEC2000 caches 1D preconditioner/norm/tcon updates every `ns4` iterations
    # (vmec_params.f: ns4=25), reusing the cached values between refreshes.
    # This materially affects the nonlinear iteration trace because the
    # Garabedian time-step control depends on ratios of the *preconditioned*
    # residual scalars.
    vmec2000_cache_valid = False
    cache_precond_diag = None
    cache_tcon = None
    cache_norms = None
    cache_rz_scale = None
    cache_l_scale = None
    cache_rz_norm = None
    cache_f_norm1 = None
    cache_prec_rz_mats = None
    cache_prec_rz_jmax = None
    cache_prec_lam_prec = None
    cache_prec_faclam = None
    cache_prec_lam_debug = None
    cache_constraint_rcon0 = None
    cache_constraint_zcon0 = None
    bcovar_update_history: list[int] = []
    iter_offset = 0

    if resume_state is not None:
        iter_offset = int(resume_state.get("iter_offset", iter_offset))
        time_step = float(resume_state.get("time_step", time_step))
        inv_tau = list(resume_state.get("inv_tau", inv_tau))
        fsq_prev = float(resume_state.get("fsq_prev", fsq_prev))
        fsq0_prev = float(resume_state.get("fsq0_prev", fsq0_prev))
        flip_sign = float(resume_state.get("flip_sign", flip_sign))
        iter1 = int(resume_state.get("iter1", iter1))
        ijacob = int(resume_state.get("ijacob", ijacob))
        bad_resets = int(resume_state.get("bad_resets", bad_resets))
        res0 = float(resume_state.get("res0", res0))
        res1 = float(resume_state.get("res1", res1))
        prev_rz_fsq = float(resume_state.get("prev_rz_fsq", prev_rz_fsq))
        bad_growth_streak = int(resume_state.get("bad_growth_streak", bad_growth_streak))
        huge_force_restart_count = int(resume_state.get("huge_force_restart_count", huge_force_restart_count))

        if "vRcc" in resume_state:
            _as_velocity = np.asarray if bool(host_update_assembly) else jnp.asarray
            vRcc = _as_velocity(resume_state["vRcc"])
            vRss = _as_velocity(resume_state.get("vRss", vRss))
            vZsc = _as_velocity(resume_state.get("vZsc", vZsc))
            vZcs = _as_velocity(resume_state.get("vZcs", vZcs))
            vLsc = _as_velocity(resume_state.get("vLsc", vLsc))
            vLcs = _as_velocity(resume_state.get("vLcs", vLcs))

        state_checkpoint = resume_state.get("state_checkpoint", state)
        vmec2000_cache_valid = bool(resume_state.get("vmec2000_cache_valid", vmec2000_cache_valid))
        cache_precond_diag = resume_state.get("cache_precond_diag", cache_precond_diag)
        cache_tcon = resume_state.get("cache_tcon", cache_tcon)
        cache_norms = resume_state.get("cache_norms", cache_norms)
        cache_rz_scale = resume_state.get("cache_rz_scale", cache_rz_scale)
        cache_l_scale = resume_state.get("cache_l_scale", cache_l_scale)
        cache_rz_norm = resume_state.get("cache_rz_norm", cache_rz_norm)
        cache_f_norm1 = resume_state.get("cache_f_norm1", cache_f_norm1)
        cache_prec_rz_mats = resume_state.get("cache_prec_rz_mats", cache_prec_rz_mats)
        cache_prec_rz_jmax = resume_state.get("cache_prec_rz_jmax", cache_prec_rz_jmax)
        cache_prec_lam_prec = resume_state.get("cache_prec_lam_prec", cache_prec_lam_prec)
        cache_prec_faclam = resume_state.get("cache_prec_faclam", cache_prec_faclam)
        cache_prec_lam_debug = resume_state.get("cache_prec_lam_debug", cache_prec_lam_debug)
        cache_constraint_rcon0 = resume_state.get("cache_constraint_rcon0", cache_constraint_rcon0)
        cache_constraint_zcon0 = resume_state.get("cache_constraint_zcon0", cache_constraint_zcon0)
        if free_boundary_enabled:
            freeb_ivac = int(resume_state.get("freeb_ivac", freeb_ivac))
            freeb_ivacskip = int(resume_state.get("freeb_ivacskip", freeb_ivacskip))
            freeb_nvacskip = max(1, int(resume_state.get("freeb_nvacskip", freeb_nvacskip)))
            freeb_nvskip0 = max(1, int(resume_state.get("freeb_nvskip0", freeb_nvskip0)))
            freeb_last_model = str(resume_state.get("freeb_model", freeb_last_model))

    def _print_axis_guess(raxis_cc, zaxis_cs) -> None:
        _print_scan_axis_guess(raxis_cc, zaxis_cs)

    def _safe_dt_from_force(
        *,
        dt_nominal: float,
        frcc,
        frss,
        fzsc,
        fzcs,
        flsc,
        flcs,
        frsc=None,
        frcs=None,
        fzcc=None,
        fzss=None,
        flcc=None,
        flss=None,
    ) -> float:
        return _safe_dt_from_force_blocks(
            dt_nominal=dt_nominal,
            max_coeff_delta_rms=max_coeff_delta_rms,
            blocks=_ForceBlocks(
                frcc,
                frss,
                fzsc,
                fzcs,
                flsc,
                flcs,
                frsc,
                frcs,
                fzcc,
                fzss,
                flcc,
                flss,
            ),
        )

    def _apply_vmec_scale_m1_precond_rhs(frzl_in: TomnspsRZL, mats: dict[str, Any]) -> TomnspsRZL:
        return _scale_m1_precond_rhs_from_mats(
            frzl_in,
            mats,
            lconm1=getattr(cfg, "lconm1", True),
            mpol=int(cfg.mpol),
            host_update_assembly=host_update_assembly,
        )

    def _pop_iteration_histories() -> None:
        def _pop(hist):
            if hist:
                hist.pop()

        for h in (
            include_edge_history,
            zero_m1_history,
            bcovar_update_history,
            w_history,
            fsqr2_history,
            fsqz2_history,
            fsql2_history,
            r00_history,
            z00_history,
            wb_history,
            wp_history,
            w_vmec_history,
            rz_norm_history,
            f_norm1_history,
            gcr2_p_history,
            gcz2_p_history,
            gcl2_p_history,
            fsq1_history,
            fsqr1_history,
            fsqz1_history,
            fsql1_history,
            min_tau_history,
            max_tau_history,
            bad_jacobian_history,
            step_history,
            dt_eff_history,
            update_rms_history,
            w_curr_history,
            w_try_history,
            w_try_ratio_history,
            restart_path_history,
            step_status_history,
            restart_reason_history,
            pre_restart_reason_history,
            time_step_history,
            res0_history,
            res1_history,
            fsq_prev_history,
            bad_growth_streak_history,
            iter1_history,
            iter2_history,
            freeb_ivac_history,
            freeb_ivacskip_history,
            freeb_full_update_history,
            grad_rms_history,
        ):
            _pop(h)

    def _maybe_dump_time_control(
        *, iter_idx: int, fsq: float, fsq0: float, res0: float, res1: float, time_step: float
    ) -> None:
        _maybe_dump_time_control_record(
            iter_idx=iter_idx,
            fsq=fsq,
            fsq0=fsq0,
            res0=res0,
            res1=res1,
            time_step=time_step,
        )

    def _dump_time_control_trace(
        *,
        stage: str,
        iter2: int,
        iter1: int,
        fsq: float,
        fsq0: float,
        res0: float,
        res1: float,
        time_step: float,
        irst: int,
    ) -> None:
        _dump_time_control_trace_record(
            stage=stage,
            iter2=iter2,
            iter1=iter1,
            fsq=fsq,
            fsq0=fsq0,
            res0=res0,
            res1=res1,
            time_step=time_step,
            irst=irst,
        )

    def _maybe_dump_checkpoint(*, iter_idx: int, fsq: float, fsq0: float, res0: float, res1: float) -> None:
        _maybe_dump_checkpoint_record(iter_idx=iter_idx, fsq=fsq, fsq0=fsq0, res0=res0, res1=res1)

    def _dump_freeb_control_trace(
        *,
        iter2: int,
        iter1: int,
        ivac: int,
        ivacskip: int,
        nvacskip: int,
        fsq_rz_prev: float,
        cached: bool,
    ) -> None:
        _dump_freeb_control_trace_record(
            iter2=iter2,
            iter1=iter1,
            ivac=ivac,
            ivacskip=ivacskip,
            nvacskip=nvacskip,
            fsq_rz_prev=fsq_rz_prev,
            cached=cached,
        )

    def _dump_freeb_axis_trace(*, iter2: int, axis_r: np.ndarray, axis_z: np.ndarray) -> None:
        _dump_freeb_axis_trace_record(iter2=iter2, axis_r=axis_r, axis_z=axis_z)

    def _dump_evolve_trace(
        *,
        iter2: int,
        iter1: int,
        stage: str,
        fsq1_val: float,
        fsq_prev_val: float,
        time_step_val: float,
        dtau_val: float,
        b1_val: float,
        fac_val: float,
        state_val: VMECState,
        vRcc_val,
        vRss_val,
        vZsc_val,
        vZcs_val,
        vLsc_val,
        vLcs_val,
        vRsc_val=None,
        vRcs_val=None,
        vZcc_val=None,
        vZss_val=None,
        vLcc_val=None,
        vLss_val=None,
        frcc_val=None,
        frss_val=None,
        fzsc_val=None,
        fzcs_val=None,
        flsc_val=None,
        flcs_val=None,
        frsc_val=None,
        frcs_val=None,
        fzcc_val=None,
        fzss_val=None,
        flcc_val=None,
        flss_val=None,
    ) -> None:
        _maybe_dump_evolve_trace_record(
            static=static,
            iter2=iter2,
            iter1=iter1,
            stage=stage,
            fsq1_val=fsq1_val,
            fsq_prev_val=fsq_prev_val,
            time_step_val=time_step_val,
            dtau_val=dtau_val,
            b1_val=b1_val,
            fac_val=fac_val,
            state_val=state_val,
            vRcc_val=vRcc_val,
            vRss_val=vRss_val,
            vZsc_val=vZsc_val,
            vZcs_val=vZcs_val,
            vLsc_val=vLsc_val,
            vLcs_val=vLcs_val,
            vRsc_val=vRsc_val,
            vRcs_val=vRcs_val,
            vZcc_val=vZcc_val,
            vZss_val=vZss_val,
            vLcc_val=vLcc_val,
            vLss_val=vLss_val,
            frcc_val=frcc_val,
            frss_val=frss_val,
            fzsc_val=fzsc_val,
            fzcs_val=fzcs_val,
            flsc_val=flsc_val,
            flcs_val=flcs_val,
            frsc_val=frsc_val,
            frcs_val=frcs_val,
            fzcc_val=fzcc_val,
            fzss_val=fzss_val,
            flcc_val=flcc_val,
            flss_val=flss_val,
        )

    # VMEC `eqsolve`: if the initial Jacobian changes sign, improve the axis
    # guess *before* the first iteration (no extra iter1). This aligns the
    # zero_m1 gating and time-control history with VMEC2000.
    t_setup_axis_reset_start = time.perf_counter() if timing_enabled else None
    if bool(vmec2000_control) and (not axis_reset_done) and bool(lmove_axis):
        try:
            t_setup_axis_force_start = time.perf_counter() if timing_enabled else None
            k0, _frzl0, _gcr2_0, _gcz2_0, _gcl2_0, _rz_scale0, _l_scale0, _norms0 = _compute_forces_iter(
                state,
                include_edge=False,
                zero_m1=jnp.asarray(1.0, dtype=jnp.asarray(state.Rcos).dtype),
                constraint_precond_diag=zero_precond_diag,
                constraint_tcon=zero_tcon,
                constraint_precond_active=jnp.asarray(False),
                constraint_tcon_active=jnp.asarray(False),
                iter_idx=None,
                iter2=1,
            )
            if timing_enabled and t_setup_axis_force_start is not None:
                try:
                    if has_jax():
                        jax.block_until_ready((_gcr2_0, _gcz2_0, _gcl2_0))
                except Exception:
                    pass
                timing_stats["setup_axis_reset_compute_forces"] += time.perf_counter() - float(t_setup_axis_force_start)
            ptau_min0, ptau_max0 = _ptau_minmax_from_k_host(k0)
            bad_jacobian_ptau = None
            if (ptau_min0 is not None) and (ptau_max0 is not None):
                min_tau_ptau = float(np.asarray(ptau_min0))
                max_tau_ptau = float(np.asarray(ptau_max0))
                bad_jacobian_ptau = (min_tau_ptau < 0.0) and (max_tau_ptau > 0.0)

            bad_jacobian_state = False
            min_tau_state = float("nan")
            max_tau_state = float("nan")
            if badjac_use_state or (bad_jacobian_ptau is None):
                jac0 = vmec_half_mesh_jacobian_from_state(
                    state=state,
                    modes=static.modes,
                    trig=trig,
                    s=s,
                    lconm1=bool(getattr(static.cfg, "lconm1", True)),
                    lthreed=bool(getattr(static.cfg, "lthreed", True)),
                    mask_even=getattr(static, "m_is_even", None),
                    mask_odd=getattr(static, "m_is_odd", None),
                )
                tau0 = jnp.asarray(jac0.tau)
                tau0_use = tau0[1:] if int(tau0.shape[0]) > 1 else tau0
                min_tau_state = float(np.asarray(jnp.min(tau0_use)))
                max_tau_state = float(np.asarray(jnp.max(tau0_use)))
                bad_jacobian_state = (min_tau_state < 0.0) and (max_tau_state > 0.0)

            axis_reset_debug = os.getenv("VMEC_JAX_AXIS_RESET_DEBUG", "").strip().lower() not in (
                "",
                "0",
                "false",
                "no",
            )
            fsq_phys0_val = None
            try:
                fsqr0 = _norms0.r1 * _norms0.fnorm * _gcr2_0
                fsqz0 = _norms0.r1 * _norms0.fnorm * _gcz2_0
                fsql0 = _norms0.fnormL * _gcl2_0
                fsq_phys0_val = float(np.asarray(fsqr0 + fsqz0 + fsql0))
            except Exception:
                fsq_phys0_val = None

            axis_reset_decision = _initial_axis_reset_decision(
                bad_jacobian_ptau=bad_jacobian_ptau,
                bad_jacobian_state=bad_jacobian_state,
                badjac_use_state=badjac_use_state,
                fsq_phys=fsq_phys0_val,
                axis_reset_fsq_min=axis_reset_fsq_min,
                force_axis_reset=force_axis_reset,
                axis_reset_always_3d=axis_reset_always_3d,
                lthreed=bool(getattr(cfg, "lthreed", True)),
            )
            bad_jacobian0 = axis_reset_decision.bad_jacobian
            if axis_reset_debug:
                try:
                    fsq_debug_val = float("nan") if fsq_phys0_val is None else float(fsq_phys0_val)
                    print(
                        "[axis_reset] fsq0="
                        f"{fsq_debug_val:.6e} "
                        f"axis_reset_fsq_min={axis_reset_fsq_min:.3e} "
                        f"badjac_ptau={bad_jacobian_ptau} badjac_state={bad_jacobian_state} "
                        f"badjac_used={bad_jacobian0}",
                        flush=True,
                    )
                except Exception:
                    pass

            force_axis_reset_init = axis_reset_decision.force_reset
            if axis_reset_decision.reset:
                if verbose and bool(vmec2000_control) and bool(verbose_vmec2000_table):
                    if bad_jacobian0 or force_axis_reset_init:
                        print(" INITIAL JACOBIAN CHANGED SIGN!", flush=True)
                    print(" TRYING TO IMPROVE INITIAL MAGNETIC AXIS GUESS", flush=True)
                state = _reset_axis_from_boundary(state, k_guess=k0, full_reset=False, refine_axis_guess=False)
                if verbose and bool(vmec2000_control) and bool(verbose_vmec2000_table):
                    if axis_reset_coeffs is not None:
                        raxis_cc, _raxis_cs, _zaxis_cc, zaxis_cs = axis_reset_coeffs
                        _print_axis_guess(raxis_cc, zaxis_cs)
                axis_reset_done = True
                ijacob = 1
                state_checkpoint = state
                vRcc, vRss, vZsc, vZcs, vLsc, vLcs = _zero_velocity_blocks_like(
                    vRcc, vRss, vZsc, vZcs, vLsc, vLcs
                )
                res0 = -1.0
                res1 = -1.0
                prev_rz_fsq = 2.0
                vmec2000_cache_valid = False
                cache_precond_diag = None
                cache_tcon = None
                cache_norms = None
                cache_rz_scale = None
                cache_l_scale = None
                cache_rz_norm = None
                cache_f_norm1 = None
                cache_prec_rz_mats = None
                cache_prec_rz_jmax = None
                cache_prec_lam_prec = None
                cache_prec_faclam = None
                cache_prec_lam_debug = None
                cache_constraint_rcon0 = None
                cache_constraint_zcon0 = None
        except Exception:
            pass
    if timing_enabled and t_setup_axis_reset_start is not None:
        timing_stats["setup_axis_reset"] += time.perf_counter() - float(t_setup_axis_reset_start)

    # Cache os.getenv calls that would otherwise be repeated every iteration
    # in the hot loop below (saves ~9 os.getenv calls × ~2144 iters = ~19k calls).
    _env_freeb_include_edge = os.getenv("VMEC_JAX_FREEB_INCLUDE_EDGE", "0").strip().lower()
    _env_force_edge_residual = os.getenv("VMEC_JAX_FORCE_EDGE_RESIDUAL", "").strip().lower()
    _env_freeb_raise = os.getenv("VMEC_JAX_FREEB_RAISE", "").strip().lower()
    _env_debug_iter = os.getenv("VMEC_JAX_DEBUG_ITER", "").strip()
    _env_dump_lam = os.getenv("VMEC_JAX_DUMP_LAM", "")
    _env_dump_lamcal = os.getenv("VMEC_JAX_DUMP_LAMCAL", "")
    _env_dump_badjac = os.getenv("VMEC_JAX_DUMP_BADJAC", "")
    _env_dump_dir = os.getenv("VMEC_JAX_DUMP_DIR", "")

    if timing_enabled:
        timing_stats["setup_total"] = time.perf_counter() - float(_solve_wall_start)
    t_iteration_loop_start = time.perf_counter() if timing_enabled else None
    last_iter2 = 0
    for it in range(max_iter):
        iter2 = it + 1 + int(iter_offset)
        last_iter2 = iter2
        converged = False
        skip_time_control = False
        force_bcovar_update = False
        time_step_report_hold: float | None = None
        freeb_turnon_applied = False
        freeb_controls_cached: tuple[int, int, int] | None = None
        while True:
            t_iteration_prepare_start = time.perf_counter() if timing_enabled else None
            iter_since_restart = iter2 - iter1
            fsq_prev_before = fsq_prev
            fsq0_prev_before = fsq0_prev
            pre_restart_reason = "none"
            if time_step_report_hold is None:
                time_step_report_hold = float(time_step)
            time_step_report = float(time_step_report_hold)
            if free_boundary_enabled:
                # Keep free-boundary cadence fixed for this `iter2` across
                # retry/restart passes in the inner while-loop.
                fsq_rz_prev = float(prev_rz_fsq) if np.isfinite(prev_rz_fsq) else 1.0
                controls_cached_before = freeb_controls_cached is not None
                if freeb_controls_cached is None:
                    freeb_ivac, freeb_ivacskip, freeb_nvacskip = _free_boundary_iter_controls_vmec(
                        iter2=int(iter2),
                        iter1=int(iter1),
                        ivac=int(freeb_ivac),
                        nvacskip=int(freeb_nvacskip),
                        nvskip0=int(freeb_nvskip0),
                        fsq_rz_prev=float(fsq_rz_prev),
                        activate_fsq=free_boundary_activate_fsq,
                    )
                    freeb_controls_cached = (
                        int(freeb_ivac),
                        int(freeb_ivacskip),
                        int(freeb_nvacskip),
                    )
                else:
                    freeb_ivac, freeb_ivacskip, freeb_nvacskip = freeb_controls_cached
                _dump_freeb_control_trace(
                    iter2=int(iter2),
                    iter1=int(iter1),
                    ivac=int(freeb_ivac),
                    ivacskip=int(freeb_ivacskip),
                    nvacskip=int(freeb_nvacskip),
                    fsq_rz_prev=float(fsq_rz_prev),
                    cached=bool(controls_cached_before),
                )
            # VMEC vacuum.f promotes ivac=0 -> 1 inside the vacuum solve.
            # Keep both values: pre-vacuum (`freeb_ivac`) for cadence/calls,
            # and post-vacuum effective (`freeb_ivac_effective`) for force/
            # residue gating in this same iteration.
            freeb_turnon_iter = bool(free_boundary_enabled) and (int(freeb_ivac) == 0) and (int(freeb_ivacskip) == 0)
            freeb_ivac_effective = int(freeb_ivac)
            if freeb_turnon_iter:
                freeb_ivac_effective = 1
            if vmec2000_control:
                # VMEC2000 `constrain_m1` logic (residue.f90):
                #   zero gcz(m=1) if (fsqz_prev < 1e-6) OR (iter2 < 2).
                fsqz_prev = float(fsqz2_history[-1]) if fsqz2_history else 1.0
                zero_m1_val = 1.0 if (iter2 < 2) or (fsqz_prev < 1.0e-6) else 0.0
            else:
                # A conservative heuristic early in a restart window.
                zero_m1_val = (
                    1.0 if (iter_since_restart < 2) or (len(fsqz2_history) and fsqz2_history[-1] < 1e-6) else 0.0
                )
            if host_update_assembly and _jnp_zero_m1_0 is not None:
                # Use pre-cached JAX scalars to avoid jnp.asarray dispatch + dtype
                # lookup every iteration (saves 2 apply_primitive calls per iter).
                zero_m1 = _jnp_zero_m1_1 if zero_m1_val > 0.5 else _jnp_zero_m1_0
            else:
                zero_m1 = jnp.asarray(zero_m1_val, dtype=jnp.asarray(state.Rcos).dtype)
            if vmec2000_control:
                # VMEC2000 keeps the core R/Z residual assembly on the
                # interior mesh; free-boundary coupling enters through the
                # dedicated edge `rbsq` terms in `forces.f`, not by enabling
                # generic edge residual rows.
                include_edge = _env_freeb_include_edge not in ("", "0", "false", "no")
            else:
                include_edge = bool(iter_since_restart < 50) and (float(prev_rz_fsq) < 1e-6)
            if track_history:
                include_edge_history.append(int(bool(include_edge)))
            # Residual transform edge handling:
            # VMEC tomnsp_mod uses jmax=ns once free-boundary vacuum is on
            # (ivac >= 1), independent of residue's `jedge` scalar gating.
            # Keep `include_edge` for scalar gating/diagnostics, but include
            # edge rows in the transform when vacuum coupling is active.
            include_edge_residual = bool(include_edge)
            if bool(free_boundary_enabled) and int(freeb_ivac_effective) >= 1:
                include_edge_residual = True
            if _env_force_edge_residual in ("1", "true", "yes"):
                include_edge_residual = True
            precond_jmax_override: int | None = None
            if bool(vmec2000_control) and bool(free_boundary_enabled) and (int(freeb_ivac_effective) >= 1):
                # VMEC scalfor: jmax=ns once free-boundary vacuum is active.
                precond_jmax_override = int(s.shape[0])
            precond_expected_jmax = (
                int(precond_jmax_override) if (precond_jmax_override is not None) else max(int(s.shape[0]) - 1, 1)
            )
            # `zero_m1` originates from host control flow, so keep the history
            # without forcing an unnecessary device synchronization.
            if track_history:
                zero_m1_history.append(int(zero_m1_val > 0.5))

            need_bcovar_update = bool(vmec2000_control) and (
                (not bool(vmec2000_cache_valid))
                or bool(force_bcovar_update)
                or ((iter2 - iter1) % k_preconditioner_update_interval == 0)
            )
            force_bcovar_update = False
            bcovar_update_history.append(int(bool(need_bcovar_update)))

            use_cached_precond = (
                bool(vmec2000_control) and bool(vmec2000_cache_valid) and (not bool(need_bcovar_update))
            )
            constraint_precond_diag = (
                cache_precond_diag if (use_cached_precond and cache_precond_diag is not None) else zero_precond_diag
            )
            # VMEC updates tcon only when refreshing the 1D preconditioner
            # blocks; between refreshes it reuses the last tcon profile.
            constraint_tcon_override = cache_tcon if (use_cached_precond and cache_tcon is not None) else zero_tcon
            if host_update_assembly and _jnp_true_bool is not None:
                # Use pre-cached bool scalars — avoids 2 jnp.asarray dispatches/iter.
                constraint_precond_active = _jnp_true_bool if use_cached_precond else _jnp_false_bool
                constraint_tcon_active = _jnp_true_bool if use_cached_precond else _jnp_false_bool
            else:
                constraint_precond_active = jnp.asarray(use_cached_precond, dtype=bool)
                constraint_tcon_active = jnp.asarray(use_cached_precond, dtype=bool)

            # Free-boundary WP2 scaffold: run/update the NESTOR-like external
            # vacuum solve and couple bsqvac on the edge slice into bcovar.
            freeb_bsqvac_half_current = None
            freeb_reused = False
            freeb_solve_time = 0.0
            freeb_sample_time = 0.0
            if bool(free_boundary_enabled and freeb_couple_edge):
                try:
                    # VMEC free-boundary path in funct3d only enters NESTOR
                    # once control is active (`ivac >= 0`), with `vacuum.f`
                    # promoting ivac=0 -> 1 internally on first turn-on.
                    if int(freeb_ivac) >= 0:
                        nestor_res, freeb_nestor_runtime = nestor_external_only_step(
                            state=state,
                            static=static,
                            ivac=int(freeb_ivac),
                            ivacskip=int(freeb_ivacskip),
                            iter_idx=int(iter2),
                            runtime=freeb_nestor_runtime,
                            extcur=tuple(getattr(static, "free_boundary_extcur", ()) or ()),
                            plascur=float(freeb_plascur),
                            external_field_provider_kind=external_field_provider_kind,
                            external_field_provider_static=external_field_provider_static,
                            external_field_provider_params=external_field_provider_params,
                        )
                        freeb_last_model = str(getattr(nestor_res, "model", "spectral_poisson_external_only"))
                        freeb_reused = bool(getattr(nestor_res, "reused", False))
                        freeb_solve_time = float(getattr(nestor_res, "solve_time_s", 0.0))
                        freeb_sample_time = float(getattr(nestor_res, "sample_time_s", 0.0))
                        diag_nestor = getattr(nestor_res, "diagnostics", None)
                        if isinstance(diag_nestor, dict):
                            freeb_last_diagnostics = dict(diag_nestor)
                            freeb_nestor_source_reused_history.append(
                                1 if bool(diag_nestor.get("source_reused", False)) else 0
                            )
                            freeb_nestor_provider_allows_source_reuse_history.append(
                                1 if bool(diag_nestor.get("provider_allows_source_reuse", False)) else 0
                            )
                            for _key, _hist in (
                                ("bnormal_rms", freeb_nestor_bnormal_rms_history),
                                ("gsource_rms", freeb_nestor_gsource_rms_history),
                                ("bsqvac_rms", freeb_nestor_bsqvac_rms_history),
                            ):
                                try:
                                    _hist.append(float(diag_nestor.get(_key, float("nan"))))
                                except Exception:
                                    _hist.append(float("nan"))
                        else:
                            freeb_nestor_source_reused_history.append(0)
                            freeb_nestor_provider_allows_source_reuse_history.append(0)
                            freeb_nestor_bnormal_rms_history.append(float("nan"))
                            freeb_nestor_gsource_rms_history.append(float("nan"))
                            freeb_nestor_bsqvac_rms_history.append(float("nan"))
                        bsqvac_edge = np.asarray(nestor_res.vac_total.bsqvac, dtype=float)
                        if (
                            bsqvac_edge.ndim == 2
                            and int(bsqvac_edge.shape[1]) == 1
                            and int(getattr(static.cfg, "nzeta", 1)) > 1
                        ):
                            bsqvac_edge = np.repeat(bsqvac_edge, int(static.cfg.nzeta), axis=1)
                        # Only the edge slice is consumed by the force kernels.
                        # Keep this as a 2D edge field so the GPU path does not
                        # re-transfer a mostly-zero `(ns, ntheta, nzeta)` array
                        # on every free-boundary iteration.
                        freeb_bsqvac_half_current = bsqvac_edge
                        if freeb_turnon_iter:
                            # VMEC promotes ivac=0 -> 1 inside vacuum.f before
                            # the same-iteration funct3d restart on turn-on.
                            freeb_ivac = 1
                            freeb_ivac_effective = 1
                            freeb_controls_cached = (
                                int(freeb_ivac),
                                int(freeb_ivacskip),
                                int(freeb_nvacskip),
                            )
                except Exception:
                    if _env_freeb_raise not in ("", "0", "false", "no"):
                        raise
                    freeb_bsqvac_half_current = None
                    freeb_reused = False
                    freeb_solve_time = 0.0
                    freeb_sample_time = 0.0

            def _freeb_bsqvac_half_for_trial_state(candidate_state: VMECState):
                """Return a non-mutating direct-provider vacuum field for trials.

                Legacy mgrid runs keep VMEC's committed ivac/ivacskip cadence.
                Direct coil providers need candidate-state sampling during
                trial/backtracking scoring so the trial boundary is not scored
                against stale pre-update vacuum source data. The scratch
                runtime returned by NESTOR is intentionally discarded so
                rejected trials cannot mutate the accepted runtime state.
                """

                if not bool(free_boundary_enabled and freeb_couple_edge):
                    return freeb_bsqvac_half_current
                if freeb_bsqvac_half_current is None:
                    return None
                provider_kind_trial = (
                    "mgrid"
                    if external_field_provider_kind is None
                    else str(external_field_provider_kind).strip().lower()
                )
                if provider_kind_trial in ("", "mgrid", "legacy_mgrid"):
                    return freeb_bsqvac_half_current
                if isinstance(external_field_provider_static, dict) and not bool(
                    external_field_provider_static.get("resample_trial_bsqvac", True)
                ):
                    return freeb_bsqvac_half_current
                if int(freeb_ivac_effective) < 1:
                    return freeb_bsqvac_half_current
                try:
                    nestor_trial, _runtime_trial = nestor_external_only_step(
                        state=candidate_state,
                        static=static,
                        ivac=1,
                        ivacskip=0,
                        iter_idx=int(iter2),
                        runtime=freeb_nestor_runtime,
                        extcur=tuple(getattr(static, "free_boundary_extcur", ()) or ()),
                        plascur=float(freeb_plascur),
                        external_field_provider_kind=external_field_provider_kind,
                        external_field_provider_static=external_field_provider_static,
                        external_field_provider_params=external_field_provider_params,
                    )
                    freeb_nestor_trial_reused_history.append(1 if bool(getattr(nestor_trial, "reused", False)) else 0)
                    freeb_nestor_trial_solve_time_history.append(float(getattr(nestor_trial, "solve_time_s", 0.0)))
                    freeb_nestor_trial_sample_time_history.append(float(getattr(nestor_trial, "sample_time_s", 0.0)))
                    freeb_nestor_trial_failed_history.append(0)
                    bsqvac_edge_trial = np.asarray(nestor_trial.vac_total.bsqvac, dtype=float)
                    if (
                        bsqvac_edge_trial.ndim == 2
                        and int(bsqvac_edge_trial.shape[1]) == 1
                        and int(getattr(static.cfg, "nzeta", 1)) > 1
                    ):
                        bsqvac_edge_trial = np.repeat(bsqvac_edge_trial, int(static.cfg.nzeta), axis=1)
                    return bsqvac_edge_trial
                except Exception:
                    freeb_nestor_trial_reused_history.append(0)
                    freeb_nestor_trial_solve_time_history.append(0.0)
                    freeb_nestor_trial_sample_time_history.append(0.0)
                    freeb_nestor_trial_failed_history.append(1)
                    if _env_freeb_raise not in ("", "0", "false", "no"):
                        raise
                    return freeb_bsqvac_half_current

            constraint_rcon0_current = None
            constraint_zcon0_current = None
            if (
                bool(vmec2000_control)
                and bool(free_boundary_enabled)
                and (cache_constraint_rcon0 is not None)
                and (cache_constraint_zcon0 is not None)
            ):
                # VMEC keeps rcon0/zcon0 as persistent baselines; once free-
                # boundary control is active, damp them by 0.9 on reuse steps.
                # The first turn-on iteration keeps the pre-turn-on baseline.
                if _free_boundary_should_damp_constraint_baseline(
                    freeb_ivac=int(freeb_ivac),
                    freeb_turnon_iter=bool(freeb_turnon_iter),
                    lthreed=bool(cfg.lthreed),
                ):
                    cache_constraint_rcon0 = 0.9 * jnp.asarray(cache_constraint_rcon0)
                    cache_constraint_zcon0 = 0.9 * jnp.asarray(cache_constraint_zcon0)
                constraint_rcon0_current = cache_constraint_rcon0
                constraint_zcon0_current = cache_constraint_zcon0

            if (
                profile_active
                and (not profile_started)
                and (profile_start_iter is not None)
                and (iter2 == profile_start_iter)
            ):
                if has_jax():
                    try:
                        Path(profile_dir).mkdir(parents=True, exist_ok=True)
                        jax.profiler.start_trace(profile_dir, create_perfetto_trace=profile_perfetto)
                        profile_started = True
                    except Exception:
                        profile_active = False

            if timing_enabled and t_iteration_prepare_start is not None:
                timing_stats["iteration_prepare"] += time.perf_counter() - float(t_iteration_prepare_start)
            t_compute_start = time.perf_counter() if timing_enabled else None
            k, frzl, gcr2, gcz2, gcl2, rz_scale, l_scale, norms_current = _compute_forces_iter(
                state,
                include_edge=bool(include_edge),
                include_edge_residual=bool(include_edge_residual),
                zero_m1=zero_m1,
                freeb_bsqvac_half=freeb_bsqvac_half_current,
                constraint_rcon0=constraint_rcon0_current,
                constraint_zcon0=constraint_zcon0_current,
                constraint_precond_diag=constraint_precond_diag,
                constraint_tcon=constraint_tcon_override,
                constraint_precond_active=constraint_precond_active,
                constraint_tcon_active=constraint_tcon_active,
                iter_idx=_iter_idx_for_dump(iter2),
                iter2=iter2,
            )
            if bool(free_boundary_enabled):
                freeb_plascur = _vmec_freeb_plascur_from_bcovar(k.bc, freeb_plascur)
                try:
                    pr1_axis = np.asarray(k.pr1_even, dtype=float)
                    pz1_axis = np.asarray(k.pz1_even, dtype=float)
                    if pr1_axis.ndim >= 3 and pz1_axis.ndim >= 3:
                        _dump_freeb_axis_trace(
                            iter2=int(iter2),
                            axis_r=np.asarray(pr1_axis[0, 0, :], dtype=float).reshape(-1),
                            axis_z=np.asarray(pz1_axis[0, 0, :], dtype=float).reshape(-1),
                        )
                except Exception:
                    pass
                if getattr(k, "constraint_rcon0", None) is not None:
                    if cache_constraint_rcon0 is None or cache_constraint_zcon0 is None:
                        # Initialize persistent VMEC-style constraint baseline.
                        cache_constraint_rcon0 = jnp.asarray(k.constraint_rcon0)
                        cache_constraint_zcon0 = jnp.asarray(k.constraint_zcon0)
            if timing_enabled:
                _record_compute_force_timing("main", t_compute_start, gcr2)
            t_residual_metrics_start = time.perf_counter() if timing_enabled else None
            norms_used = (
                cache_norms
                if (bool(vmec2000_control) and bool(vmec2000_cache_valid) and (not bool(need_bcovar_update)))
                else norms_current
            )
            if host_update_assembly:
                # NumPy path: gcr2/gcz2/gcl2 already synced by block_until_ready above.
                # float() on synced JAX scalars is fast (no blocking). Avoids 5 JAX dispatches.
                _gcr2_f = float(gcr2)
                _gcz2_f = float(gcz2)
                _gcl2_f = float(gcl2)
                _fnorm_f = float(norms_used.fnorm)
                _fnormL_f = float(norms_used.fnormL)
                _r1_f = float(norms_used.r1)
                fsqr = _r1_f * _fnorm_f * _gcr2_f
                fsqz = _r1_f * _fnorm_f * _gcz2_f
                fsql = _fnormL_f * _gcl2_f
            else:
                fsqr = norms_used.r1 * norms_used.fnorm * gcr2
                fsqz = norms_used.r1 * norms_used.fnorm * gcz2
                fsql = norms_used.fnormL * gcl2
            debug_iter_env = _env_debug_iter
            _maybe_print_nonscan_state_debug(
                debug_iter_env=debug_iter_env,
                iter2=int(iter2),
                state=state,
                state_checkpoint=state_checkpoint,
                gcr2=gcr2,
                gcz2=gcz2,
                gcl2=gcl2,
                norms_used=norms_used,
                print_fn=print,
            )
            if bool(vmec2000_control) and bool(vmec2000_cache_valid) and (not bool(need_bcovar_update)):
                rz_scale = cache_rz_scale
                l_scale = cache_l_scale
            if bool(vmec2000_control) and bool(need_bcovar_update):
                if constraint_tcon0 is None or float(constraint_tcon0) == 0.0:
                    cache_precond_diag = None
                    cache_tcon = jnp.zeros((int(s.shape[0]),), dtype=jnp.asarray(state.Rcos).dtype)
                else:
                    from .vmec_constraints import precondn_diag_axd1_from_bcovar

                    if host_update_assembly and (not _tree_has_tracer(k)) and (not _tree_has_tracer(s)):
                        from .vmec_numpy_forces import _numpy_module_patch as _hot_numpy_patch

                        with _hot_numpy_patch():
                            ard1, azd1 = precondn_diag_axd1_from_bcovar(
                                trig=trig,
                                s=s,
                                bsq=k.bc.bsq,
                                r12=k.bc.jac.r12,
                                sqrtg=k.bc.jac.sqrtg,
                                ru12=k.bc.jac.ru12,
                                zu12=k.bc.jac.zu12,
                            )
                    else:
                        ard1, azd1 = precondn_diag_axd1_from_bcovar(
                            trig=trig,
                            s=s,
                            bsq=k.bc.bsq,
                            r12=k.bc.jac.r12,
                            sqrtg=k.bc.jac.sqrtg,
                            ru12=k.bc.jac.ru12,
                            zu12=k.bc.jac.zu12,
                        )
                    cache_precond_diag = (ard1, azd1)
                    cache_tcon = np.asarray(k.tcon) if host_update_assembly else jnp.asarray(k.tcon)
                cache_norms = norms_used
                cache_rz_scale = rz_scale
                cache_l_scale = l_scale
                if host_update_assembly:
                    # NumPy path: avoids JAX dispatch + XLA blocking for fnorm1.
                    cache_rz_norm = _rz_norm_np(state)  # Python float
                    cache_f_norm1 = (1.0 / cache_rz_norm) if cache_rz_norm != 0.0 else float("inf")
                else:
                    cache_rz_norm = _rz_norm(state)
                    cache_f_norm1 = jnp.where(
                        jnp.asarray(cache_rz_norm) != 0.0,
                        1.0 / jnp.asarray(cache_rz_norm),
                        jnp.asarray(float("inf"), dtype=jnp.asarray(cache_rz_norm).dtype),
                    )
                if not bool(cfg.lasym):
                    cache_prec_lam_prec = _lambda_preconditioner(k.bc)
                    mats, _jmin, jmax = _rz_preconditioner_matrices_local(
                        bc=k.bc,
                        k=k,
                        jmax_override=precond_jmax_override,
                        use_precomputed=preconditioner_use_precomputed_tridi_policy,
                        use_lax_tridi=preconditioner_use_lax_tridi_policy,
                    )
                    cache_prec_rz_mats = mats
                    cache_prec_rz_jmax = None if _tree_has_tracer(k) else int(jmax)
                vmec2000_cache_valid = True
            if host_update_assembly:
                # fsqr/fsqz/fsql are already Python floats from the NumPy path above.
                fsqr_f, fsqz_f, fsql_f = fsqr, fsqz, fsql
            else:
                fsqr_f, fsqz_f, fsql_f = _device_get_floats(fsqr, fsqz, fsql)
            if bool(free_boundary_enabled) and bool(freeb_turnon_iter) and (not bool(freeb_turnon_applied)):
                # VMEC restarts funct3d immediately after the first
                # free-boundary turn-on solve, keeping the cached ns4 blocks
                # intact across the same-iteration retry.
                state = state_checkpoint
                (
                    vRcc,
                    vRss,
                    vZsc,
                    vZcs,
                    vLsc,
                    vLcs,
                    vRsc,
                    vRcs,
                    vZcc,
                    vZss,
                    vLcc,
                    vLss,
                ) = _zero_velocity_blocks_like(vRcc, vRss, vZsc, vZcs, vLsc, vLcs, vRsc, vRcs, vZcc, vZss, vLcc, vLss)
                time_step_report_hold = float(time_step)
                ijacob += 1
                if _free_boundary_turnon_resets_iter1_immediately(
                    lthreed=bool(cfg.lthreed),
                    lasym=bool(cfg.lasym),
                ):
                    iter1 = int(iter2)
                bad_growth_streak = 0
                inv_tau = [0.15 / max(float(time_step), 1e-12)] * k_ndamp
                freeb_turnon_applied = True
            fsq0_curr = fsqr_f + fsqz_f + fsql_f
            prev_rz_fsq_before = prev_rz_fsq
            prev_rz_fsq = _free_boundary_prev_rz_fsq_next(
                prev_fsq_before=prev_rz_fsq_before,
                fsq_rz_curr=fsqr_f + fsqz_f,
                turnon_restart=bool(free_boundary_enabled) and bool(freeb_turnon_iter) and bool(freeb_turnon_applied),
                preserve_turnon_restart=bool(free_boundary_enabled) and bool(cfg.lthreed),
            )

            w_history.append(fsq0_curr)
            fsqr2_history.append(fsqr_f)
            fsqz2_history.append(fsqz_f)
            fsql2_history.append(fsql_f)
            # VMEC printout uses r00 = r1(1,0): axis R at theta=0, zeta=0,
            # evaluated in real space after scalxc (see funct3d.f).
            # For parity diagnostics, sample these scalars on VMEC's screen cadence.
            sample_vmec = bool(vmec2000_control) and _should_sample_vmec2000(int(iter2), int(max_iter))
            need_scalar = bool(sample_vmec) or (bool(verbose) and (not bool(vmec2000_control)))
            if need_scalar:
                if host_update_assembly and (not _tree_has_tracer(k)):
                    try:
                        r00_val = float(np.asarray(k.pr1_even)[0, 0, 0])
                        z00_val = float(np.asarray(k.pz1_even)[0, 0, 0]) if bool(cfg.lasym) else 0.0
                    except Exception:
                        if not np.any(m0_mask):
                            r00_val = float("nan")
                            z00_val = float("nan")
                        else:
                            r00_val = float(np.sum(np.asarray(state.Rcos)[0, m0_mask]))
                            z00_val = float(np.sum(np.asarray(state.Zcos)[0, m0_mask])) if bool(cfg.lasym) else 0.0
                    wb_val = float(np.asarray(norms_current.wb))
                    wp_val = float(np.asarray(norms_current.wp))
                else:
                    try:
                        r00_j = jnp.asarray(k.pr1_even)[0, 0, 0]
                        if bool(cfg.lasym):
                            z00_j = jnp.asarray(k.pz1_even)[0, 0, 0]
                        else:
                            z00_j = jnp.asarray(0.0, dtype=jnp.asarray(r00_j).dtype)
                    except Exception:
                        if not np.any(m0_mask):
                            r00_j = jnp.asarray(float("nan"))
                            z00_j = jnp.asarray(float("nan"))
                        else:
                            r00_j = jnp.sum(jnp.asarray(state.Rcos)[0, m0_mask])
                            if bool(cfg.lasym):
                                z00_j = jnp.sum(jnp.asarray(state.Zcos)[0, m0_mask])
                            else:
                                z00_j = jnp.asarray(0.0, dtype=jnp.asarray(r00_j).dtype)
                    # `norms_used` may be cached (VMEC2000 `ns4=25` behavior), but
                    # `norms_current` already reflects the current bcovar state and
                    # therefore matches VMEC's printed wb/wp without recomputing.
                    wb_j = jnp.asarray(norms_current.wb)
                    wp_j = jnp.asarray(norms_current.wp)
                    r00_val, z00_val, wb_val, wp_val = _device_get_floats(r00_j, z00_j, wb_j, wp_j)
                if bool(vmec2000_control):
                    # Match VMEC's printed precision (E11.3) for parity checks.
                    r00_val = float(f"{float(r00_val):.3E}")
                    z00_val = float(f"{float(z00_val):.3E}")
            else:
                r00_val = r00_last
                z00_val = z00_last
                wb_val = wb_last
                wp_val = wp_last
            r00_last = float(r00_val)
            z00_last = float(z00_val)
            wb_last = float(wb_val)
            wp_last = float(wp_val)
            w_vmec_last = (wb_last + wp_last / (gamma - 1.0)) * float(TWOPI * TWOPI)
            if track_history:
                r00_history.append(r00_last)
                z00_history.append(z00_last)
                wb_history.append(wb_last)
                wp_history.append(wp_last)
                w_vmec_history.append(w_vmec_last)

            if verbose and (not (bool(vmec2000_control) and bool(verbose_vmec2000_table))):
                print(
                    f"[solve_fixed_boundary_residual_iter] iter={it:03d} fsqr={fsqr_f:.3e} fsqz={fsqz_f:.3e} "
                    f"fsql={fsql_f:.3e} include_edge={include_edge}",
                    flush=True,
                )
            # Defer convergence exit until after preconditioned diagnostics are
            # computed for this iteration, so fsqr1/fsqz1/fsql1 histories and
            # VMEC-style tables remain length-aligned.
            converged_physical = _converged_residuals_host(fsqr=fsqr_f, fsqz=fsqz_f, fsql=fsql_f)
            accepted_control_ptau_payload: tuple[Any, Any, Any] | None = None
            fuse_accepted_control_ptau = (
                bool(free_boundary_enabled)
                and bool(direct_free_boundary_provider)
                and (not bool(converged_physical))
                and (bool(reference_mode) or bool(vmec2000_control))
                and (not bool(host_update_assembly))
                and (not bool(badjac_use_state))
                and (not bool(dump_ptau_state))
                and os.getenv("VMEC_JAX_DUMP_PTAU", "") in ("", "0")
                and jax.default_backend() != "cpu"
            )
            accepted_control_ptau_arrays = _accepted_control_ptau_arrays(k) if fuse_accepted_control_ptau else None

            # Precondition forces.
            if timing_enabled and t_residual_metrics_start is not None:
                timing_stats["iteration_residual_metrics"] += time.perf_counter() - float(t_residual_metrics_start)
            t_precond_start = time.perf_counter() if timing_enabled else None
            frzl_lam_pre = None
            preconditioner_outputs_scaled = False
            preconditioner_fsq1_ready = False
            use_fused_precond_output_scaling = (not bool(host_update_assembly)) and jax.default_backend() != "cpu"
            if bool(vmec2000_control) and bool(cfg.lthreed):
                from .preconditioner_1d_jax import (
                    rz_preconditioner_matrices_reassemble,
                )

                precond_traced = _tree_has_tracer(k)
                need_lam_prec = _env_dump_lam not in ("", "0")
                need_lamcal = _env_dump_lamcal not in ("", "0")
                need_prec_reassemble = (
                    (not precond_traced)
                    and (cache_prec_rz_jmax is not None)
                    and (int(cache_prec_rz_jmax) != int(precond_expected_jmax))
                    and _can_reassemble_precond_mats(cache_prec_rz_mats)
                )
                need_prec_refresh = (
                    precond_traced
                    or (not bool(vmec2000_cache_valid))
                    or (cache_prec_lam_prec is None)
                    or (cache_prec_rz_mats is None)
                    or (cache_prec_rz_jmax is None)
                    or bool(need_bcovar_update)
                    or (
                        (cache_prec_rz_jmax is not None)
                        and (int(cache_prec_rz_jmax) != int(precond_expected_jmax))
                        and (not bool(need_prec_reassemble))
                    )
                )
                if need_prec_refresh:
                    t_prec_refresh_start = time.perf_counter() if timing_enabled else None
                    if need_lamcal:
                        if need_lam_prec:
                            lam_prec, faclam_dump, lam_debug = _lambda_preconditioner(
                                k.bc, return_faclam=True, return_debug=True
                            )
                        else:
                            lam_prec, lam_debug = _lambda_preconditioner(k.bc, return_debug=True)
                            faclam_dump = None
                    else:
                        if need_lam_prec:
                            lam_prec, faclam_dump = _lambda_preconditioner(k.bc, return_faclam=True)
                        else:
                            lam_prec = _lambda_preconditioner(k.bc)
                            faclam_dump = None
                        lam_debug = None
                    mats, _jmin, jmax = _rz_preconditioner_matrices_local(
                        bc=k.bc,
                        k=k,
                        jmax_override=precond_jmax_override,
                        use_precomputed=preconditioner_use_precomputed_tridi_policy,
                        use_lax_tridi=preconditioner_use_lax_tridi_policy,
                    )
                    cache_prec_lam_prec = lam_prec
                    cache_prec_faclam = faclam_dump
                    cache_prec_lam_debug = lam_debug
                    cache_prec_rz_mats = mats
                    cache_prec_rz_jmax = None if precond_traced else int(jmax)
                    if timing_enabled and t_prec_refresh_start is not None:
                        try:
                            if has_jax():
                                jax.block_until_ready(lam_prec)
                        except Exception:
                            pass
                        timing_stats["precond_refresh"] += time.perf_counter() - float(t_prec_refresh_start)
                else:
                    lam_prec = cache_prec_lam_prec
                    faclam_dump = cache_prec_faclam if need_lam_prec else None
                    lam_debug = cache_prec_lam_debug if need_lamcal else None
                    if bool(need_prec_reassemble):
                        mats, _jmin, jmax = rz_preconditioner_matrices_reassemble(
                            mats=cache_prec_rz_mats,
                            cfg=cfg,
                            jmax_override=precond_jmax_override,
                        )
                        cache_prec_rz_mats = mats
                        cache_prec_rz_jmax = None if precond_traced else int(jmax)
                    else:
                        mats = cache_prec_rz_mats
                        jmax = cache_prec_rz_jmax
                _maybe_dump_lam_prec(lam_prec=lam_prec, faclam=faclam_dump, static=static, iter_idx=int(iter2))
                if not precond_traced:
                    _maybe_dump_precond_mats(
                        mats=mats,
                        static=static,
                        iter_idx=int(iter2),
                        jmax=int(jmax),
                        used_cache=(not bool(need_prec_refresh)),
                    )
                if lam_debug is not None:
                    _maybe_dump_lamcal(lam_debug=lam_debug, static=static, iter_idx=int(iter2))
                t_precond_apply_start = time.perf_counter() if timing_detail_enabled else None
                use_apply_payload_fusion = (
                    bool(use_fused_precond_output_scaling)
                    and need_lam_prec is False
                    and need_lamcal is False
                )
                frzl_rhs = _apply_vmec_scale_m1_precond_rhs(frzl, mats)
                if use_apply_payload_fusion:
                    if (
                        bool(vmec2000_control)
                        and bool(vmec2000_cache_valid)
                        and (not bool(need_bcovar_update))
                        and (cache_rz_norm is not None)
                        and (cache_f_norm1 is not None)
                    ):
                        f_norm1 = jnp.asarray(cache_f_norm1)
                    else:
                        rz_norm = _rz_norm(state)
                        f_norm1 = jnp.where(
                            rz_norm != 0.0,
                            1.0 / rz_norm,
                            jnp.asarray(float("inf"), dtype=rz_norm.dtype),
                        )
                    _precond_payload = _preconditioner_apply_payload_fused(
                        frzl_in=frzl_rhs,
                        mats=mats,
                        jmax=jmax,
                        cfg=cfg,
                        lam_prec=lam_prec,
                        w_mode_mn=w_mode_mn,
                        lambda_update_scale_j=lambda_update_scale_j,
                        f_norm1=f_norm1,
                        delta_s=delta_s,
                        s=s,
                        use_precomputed=preconditioner_use_precomputed_tridi_policy,
                        use_lax_tridi=preconditioner_use_lax_tridi_policy,
                        apply_lambda_update_scale=(lambda_update_scale != 1.0),
                        vmec2000_control=bool(vmec2000_control),
                        lconm1=bool(getattr(static.cfg, "lconm1", True)),
                        include_control_ptau=accepted_control_ptau_arrays is not None,
                        control_ptau_arrays=accepted_control_ptau_arrays,
                        control_ptau_pshalf=_ptau_pshalf_jax,
                        control_ptau_ohs=_ptau_ohs_jax,
                    )
                    if len(_precond_payload) == 4:
                        (
                            _precond_pre_blocks,
                            _precond_update_blocks,
                            _precond_diag,
                            accepted_control_ptau_payload,
                        ) = _precond_payload
                    else:
                        _precond_pre_blocks, _precond_update_blocks, _precond_diag = _precond_payload
                    (
                        (frcc, frss, fzsc, fzcs, flsc, flcs, frsc, frcs, fzcc, fzss, flcc, flss),
                        (
                            frcc_u,
                            frss_u,
                            fzsc_u,
                            fzcs_u,
                            flsc_u,
                            flcs_u,
                            frsc_u,
                            frcs_u,
                            fzcc_u,
                            fzss_u,
                            flcc_u,
                            flss_u,
                        ),
                        (gcr2_p, gcz2_p, gcl2_p, fsqr1_safe, fsqz1_safe, fsql1_safe, fsq1_safe),
                    ) = (_precond_pre_blocks, _precond_update_blocks, _precond_diag)
                    fsqr1 = fsqr1_safe
                    fsqz1 = fsqz1_safe
                    fsql1 = fsql1_safe
                    preconditioner_outputs_scaled = True
                    preconditioner_fsq1_ready = True
                else:
                    frzl_rz = _rz_preconditioner_apply_local(
                        frzl_in=frzl_rhs,
                        mats=mats,
                        jmax=jmax,
                        use_precomputed=preconditioner_use_precomputed_tridi_policy,
                        use_lax_tridi=preconditioner_use_lax_tridi_policy,
                    )
                    frzl_lam_pre = frzl_rz
                if (not use_apply_payload_fusion) and host_update_assembly:
                    # NumPy path: avoids ~15 JAX dispatches (jnp.asarray, zeros_like, mul).
                    # Asymmetric (lasym) components default to None — the downstream
                    # mode-diag scaling uses _z (pre-allocated zeros) for None entries,
                    # avoiding 6 np.zeros_like allocations per iteration (~0.5s saving).
                    (frcc, frss, fzsc, fzcs, flsc, flcs, frsc, frcs, fzcc, fzss, flcc, flss) = (
                        _preconditioner_output_blocks_np(frzl_rz=frzl_rz, lam_prec=lam_prec)
                    )
                elif (not use_apply_payload_fusion) and use_fused_precond_output_scaling:
                    if (
                        bool(vmec2000_control)
                        and bool(vmec2000_cache_valid)
                        and (not bool(need_bcovar_update))
                        and (cache_rz_norm is not None)
                        and (cache_f_norm1 is not None)
                    ):
                        rz_norm = jnp.asarray(cache_rz_norm)
                        f_norm1 = jnp.asarray(cache_f_norm1)
                    else:
                        rz_norm = _rz_norm(state)
                        f_norm1 = jnp.where(
                            rz_norm != 0.0,
                            1.0 / rz_norm,
                            jnp.asarray(float("inf"), dtype=rz_norm.dtype),
                        )
                    payload_outputs = _preconditioner_output_payload_jit(
                        apply_lambda_update_scale=(lambda_update_scale != 1.0),
                        vmec2000_control=bool(vmec2000_control),
                        lconm1=bool(getattr(static.cfg, "lconm1", True)),
                    )
                    (
                        (frcc, frss, fzsc, fzcs, flsc, flcs, frsc, frcs, fzcc, fzss, flcc, flss),
                        (
                            frcc_u,
                            frss_u,
                            fzsc_u,
                            fzcs_u,
                            flsc_u,
                            flcs_u,
                            frsc_u,
                            frcs_u,
                            fzcc_u,
                            fzss_u,
                            flcc_u,
                            flss_u,
                        ),
                        (gcr2_p, gcz2_p, gcl2_p, fsqr1_safe, fsqz1_safe, fsql1_safe, fsq1_safe),
                    ) = payload_outputs(frzl_rz, lam_prec, w_mode_mn, lambda_update_scale_j, f_norm1, delta_s, s)
                    fsqr1 = fsqr1_safe
                    fsqz1 = fsqz1_safe
                    fsql1 = fsql1_safe
                    preconditioner_outputs_scaled = True
                    preconditioner_fsq1_ready = True
                elif not use_apply_payload_fusion:
                    frcc = jnp.asarray(frzl_rz.frcc)
                    frss = frzl_rz.frss
                    fzsc = jnp.asarray(frzl_rz.fzsc)
                    fzcs = frzl_rz.fzcs
                    flsc = jnp.asarray(frzl_rz.flsc) * jnp.asarray(lam_prec)
                    flcs = None if frzl_rz.flcs is None else (jnp.asarray(frzl_rz.flcs) * jnp.asarray(lam_prec))
                    frsc = jnp.zeros_like(frcc)
                    frcs = jnp.zeros_like(frcc)
                    fzcc = jnp.zeros_like(fzsc)
                    fzss = jnp.zeros_like(fzsc)
                    flcc = jnp.zeros_like(flsc)
                    flss = jnp.zeros_like(flsc)
                    if getattr(frzl_rz, "frsc", None) is not None:
                        frsc = jnp.asarray(frzl_rz.frsc)
                    if getattr(frzl_rz, "frcs", None) is not None:
                        frcs = jnp.asarray(frzl_rz.frcs)
                    if getattr(frzl_rz, "fzcc", None) is not None:
                        fzcc = jnp.asarray(frzl_rz.fzcc)
                    if getattr(frzl_rz, "fzss", None) is not None:
                        fzss = jnp.asarray(frzl_rz.fzss)
                    if getattr(frzl_rz, "flcc", None) is not None:
                        flcc = jnp.asarray(frzl_rz.flcc) * jnp.asarray(lam_prec)
                    if getattr(frzl_rz, "flss", None) is not None:
                        flss = jnp.asarray(frzl_rz.flss) * jnp.asarray(lam_prec)
                if timing_detail_enabled and t_precond_apply_start is not None:
                    try:
                        if has_jax():
                            jax.block_until_ready(flsc)
                    except Exception:
                        pass
                    timing_stats["precond_apply"] += time.perf_counter() - float(t_precond_apply_start)
            elif not bool(cfg.lthreed):
                from .preconditioner_1d_jax import (
                    rz_preconditioner_matrices_reassemble,
                )

                precond_traced = _tree_has_tracer(k)
                need_lam_prec = _env_dump_lam not in ("", "0")
                need_lamcal = _env_dump_lamcal not in ("", "0")
                need_prec_reassemble = (
                    (not precond_traced)
                    and (cache_prec_rz_jmax is not None)
                    and (int(cache_prec_rz_jmax) != int(precond_expected_jmax))
                    and _can_reassemble_precond_mats(cache_prec_rz_mats)
                )
                need_prec_refresh = (
                    precond_traced
                    or (not bool(vmec2000_cache_valid))
                    or (cache_prec_lam_prec is None)
                    or (cache_prec_rz_mats is None)
                    or (cache_prec_rz_jmax is None)
                    or bool(need_bcovar_update)
                    or (
                        (cache_prec_rz_jmax is not None)
                        and (int(cache_prec_rz_jmax) != int(precond_expected_jmax))
                        and (not bool(need_prec_reassemble))
                    )
                )
                if need_prec_refresh:
                    t_prec_refresh_start = time.perf_counter() if timing_enabled else None
                    if need_lamcal:
                        if need_lam_prec:
                            lam_prec, faclam_dump, lam_debug = _lambda_preconditioner(
                                k.bc, return_faclam=True, return_debug=True
                            )
                        else:
                            lam_prec, lam_debug = _lambda_preconditioner(k.bc, return_debug=True)
                            faclam_dump = None
                    else:
                        if need_lam_prec:
                            lam_prec, faclam_dump = _lambda_preconditioner(k.bc, return_faclam=True)
                        else:
                            lam_prec = _lambda_preconditioner(k.bc)
                            faclam_dump = None
                        lam_debug = None
                    mats, _jmin, jmax = _rz_preconditioner_matrices_local(
                        bc=k.bc,
                        k=k,
                        jmax_override=precond_jmax_override,
                        use_precomputed=preconditioner_use_precomputed_tridi_policy,
                        use_lax_tridi=preconditioner_use_lax_tridi_policy,
                    )
                    cache_prec_lam_prec = lam_prec
                    cache_prec_faclam = faclam_dump
                    cache_prec_lam_debug = lam_debug
                    cache_prec_rz_mats = mats
                    cache_prec_rz_jmax = None if precond_traced else int(jmax)
                    if timing_enabled and t_prec_refresh_start is not None:
                        try:
                            if has_jax():
                                jax.block_until_ready(lam_prec)
                        except Exception:
                            pass
                        timing_stats["precond_refresh"] += time.perf_counter() - float(t_prec_refresh_start)
                else:
                    lam_prec = cache_prec_lam_prec
                    faclam_dump = cache_prec_faclam if need_lam_prec else None
                    lam_debug = cache_prec_lam_debug if need_lamcal else None
                    if bool(need_prec_reassemble):
                        mats, _jmin, jmax = rz_preconditioner_matrices_reassemble(
                            mats=cache_prec_rz_mats,
                            cfg=cfg,
                            jmax_override=precond_jmax_override,
                        )
                        cache_prec_rz_mats = mats
                        cache_prec_rz_jmax = None if precond_traced else int(jmax)
                    else:
                        mats = cache_prec_rz_mats
                        jmax = cache_prec_rz_jmax
                _maybe_dump_lam_prec(lam_prec=lam_prec, faclam=faclam_dump, static=static, iter_idx=int(iter2))
                if not precond_traced:
                    _maybe_dump_precond_mats(
                        mats=mats,
                        static=static,
                        iter_idx=int(iter2),
                        jmax=int(jmax),
                        used_cache=(not bool(need_prec_refresh)),
                    )
                if lam_debug is not None:
                    _maybe_dump_lamcal(lam_debug=lam_debug, static=static, iter_idx=int(iter2))
                t_precond_apply_start = time.perf_counter() if timing_detail_enabled else None
                use_apply_payload_fusion = (
                    bool(use_fused_precond_output_scaling)
                    and need_lam_prec is False
                    and need_lamcal is False
                )
                frzl_rhs = _apply_vmec_scale_m1_precond_rhs(frzl, mats) if bool(getattr(cfg, "lasym", False)) else frzl
                if use_apply_payload_fusion:
                    if (
                        bool(vmec2000_control)
                        and bool(vmec2000_cache_valid)
                        and (not bool(need_bcovar_update))
                        and (cache_rz_norm is not None)
                        and (cache_f_norm1 is not None)
                    ):
                        f_norm1 = jnp.asarray(cache_f_norm1)
                    else:
                        rz_norm = _rz_norm(state)
                        f_norm1 = jnp.where(
                            rz_norm != 0.0,
                            1.0 / rz_norm,
                            jnp.asarray(float("inf"), dtype=rz_norm.dtype),
                        )
                    _precond_payload = _preconditioner_apply_payload_fused(
                        frzl_in=frzl_rhs,
                        mats=mats,
                        jmax=jmax,
                        cfg=cfg,
                        lam_prec=lam_prec,
                        w_mode_mn=w_mode_mn,
                        lambda_update_scale_j=lambda_update_scale_j,
                        f_norm1=f_norm1,
                        delta_s=delta_s,
                        s=s,
                        use_precomputed=preconditioner_use_precomputed_tridi_policy,
                        use_lax_tridi=preconditioner_use_lax_tridi_policy,
                        apply_lambda_update_scale=(lambda_update_scale != 1.0),
                        vmec2000_control=bool(vmec2000_control),
                        lconm1=bool(getattr(static.cfg, "lconm1", True)),
                        include_control_ptau=accepted_control_ptau_arrays is not None,
                        control_ptau_arrays=accepted_control_ptau_arrays,
                        control_ptau_pshalf=_ptau_pshalf_jax,
                        control_ptau_ohs=_ptau_ohs_jax,
                    )
                    if len(_precond_payload) == 4:
                        (
                            _precond_pre_blocks,
                            _precond_update_blocks,
                            _precond_diag,
                            accepted_control_ptau_payload,
                        ) = _precond_payload
                    else:
                        _precond_pre_blocks, _precond_update_blocks, _precond_diag = _precond_payload
                    (
                        (frcc, frss, fzsc, fzcs, flsc, flcs, frsc, frcs, fzcc, fzss, flcc, flss),
                        (
                            frcc_u,
                            frss_u,
                            fzsc_u,
                            fzcs_u,
                            flsc_u,
                            flcs_u,
                            frsc_u,
                            frcs_u,
                            fzcc_u,
                            fzss_u,
                            flcc_u,
                            flss_u,
                        ),
                        (gcr2_p, gcz2_p, gcl2_p, fsqr1_safe, fsqz1_safe, fsql1_safe, fsq1_safe),
                    ) = (_precond_pre_blocks, _precond_update_blocks, _precond_diag)
                    fsqr1 = fsqr1_safe
                    fsqz1 = fsqz1_safe
                    fsql1 = fsql1_safe
                    preconditioner_outputs_scaled = True
                    preconditioner_fsq1_ready = True
                else:
                    frzl_rz = _rz_preconditioner_apply_local(
                        frzl_in=frzl_rhs,
                        mats=mats,
                        jmax=jmax,
                        use_precomputed=preconditioner_use_precomputed_tridi_policy,
                        use_lax_tridi=preconditioner_use_lax_tridi_policy,
                    )
                    frzl_lam_pre = frzl_rz
                if (not use_apply_payload_fusion) and host_update_assembly:
                    # NumPy path: avoids ~15 JAX dispatches (jnp.asarray, zeros_like, mul).
                    # Asymmetric (lasym) components default to None — the downstream
                    # mode-diag scaling uses _z (pre-allocated zeros) for None entries,
                    # avoiding 6 np.zeros_like allocations per iteration (~0.5s saving).
                    (frcc, frss, fzsc, fzcs, flsc, flcs, frsc, frcs, fzcc, fzss, flcc, flss) = (
                        _preconditioner_output_blocks_np(frzl_rz=frzl_rz, lam_prec=lam_prec)
                    )
                elif (not use_apply_payload_fusion) and use_fused_precond_output_scaling:
                    if (
                        bool(vmec2000_control)
                        and bool(vmec2000_cache_valid)
                        and (not bool(need_bcovar_update))
                        and (cache_rz_norm is not None)
                        and (cache_f_norm1 is not None)
                    ):
                        rz_norm = jnp.asarray(cache_rz_norm)
                        f_norm1 = jnp.asarray(cache_f_norm1)
                    else:
                        rz_norm = _rz_norm(state)
                        f_norm1 = jnp.where(
                            rz_norm != 0.0,
                            1.0 / rz_norm,
                            jnp.asarray(float("inf"), dtype=rz_norm.dtype),
                        )
                    payload_outputs = _preconditioner_output_payload_jit(
                        apply_lambda_update_scale=(lambda_update_scale != 1.0),
                        vmec2000_control=bool(vmec2000_control),
                        lconm1=bool(getattr(static.cfg, "lconm1", True)),
                    )
                    (
                        (frcc, frss, fzsc, fzcs, flsc, flcs, frsc, frcs, fzcc, fzss, flcc, flss),
                        (
                            frcc_u,
                            frss_u,
                            fzsc_u,
                            fzcs_u,
                            flsc_u,
                            flcs_u,
                            frsc_u,
                            frcs_u,
                            fzcc_u,
                            fzss_u,
                            flcc_u,
                            flss_u,
                        ),
                        (gcr2_p, gcz2_p, gcl2_p, fsqr1_safe, fsqz1_safe, fsql1_safe, fsq1_safe),
                    ) = payload_outputs(frzl_rz, lam_prec, w_mode_mn, lambda_update_scale_j, f_norm1, delta_s, s)
                    fsqr1 = fsqr1_safe
                    fsqz1 = fsqz1_safe
                    fsql1 = fsql1_safe
                    preconditioner_outputs_scaled = True
                    preconditioner_fsq1_ready = True
                elif not use_apply_payload_fusion:
                    frcc = jnp.asarray(frzl_rz.frcc)
                    frss = frzl_rz.frss
                    fzsc = jnp.asarray(frzl_rz.fzsc)
                    fzcs = frzl_rz.fzcs
                    flsc = jnp.asarray(frzl_rz.flsc) * jnp.asarray(lam_prec)
                    flcs = None if frzl_rz.flcs is None else (jnp.asarray(frzl_rz.flcs) * jnp.asarray(lam_prec))
                    frsc = jnp.zeros_like(frcc)
                    frcs = jnp.zeros_like(frcc)
                    fzcc = jnp.zeros_like(fzsc)
                    fzss = jnp.zeros_like(fzsc)
                    flcc = jnp.zeros_like(flsc)
                    flss = jnp.zeros_like(flsc)
                    if getattr(frzl_rz, "frsc", None) is not None:
                        frsc = jnp.asarray(frzl_rz.frsc)
                    if getattr(frzl_rz, "frcs", None) is not None:
                        frcs = jnp.asarray(frzl_rz.frcs)
                    if getattr(frzl_rz, "fzcc", None) is not None:
                        fzcc = jnp.asarray(frzl_rz.fzcc)
                    if getattr(frzl_rz, "fzss", None) is not None:
                        fzss = jnp.asarray(frzl_rz.fzss)
                    if getattr(frzl_rz, "flcc", None) is not None:
                        flcc = jnp.asarray(frzl_rz.flcc) * jnp.asarray(lam_prec)
                    if getattr(frzl_rz, "flss", None) is not None:
                        flss = jnp.asarray(frzl_rz.flss) * jnp.asarray(lam_prec)
                if timing_detail_enabled and t_precond_apply_start is not None:
                    try:
                        if has_jax():
                            jax.block_until_ready(flsc)
                    except Exception:
                        pass
                    timing_stats["precond_apply"] += time.perf_counter() - float(t_precond_apply_start)
            else:
                t_precond_apply_start = time.perf_counter() if timing_detail_enabled else None
                frcc = _apply_radial_tridi(frzl.frcc * rz_scale[:, None, None], precond_radial_alpha)
                frss = (
                    _apply_radial_tridi(frzl.frss * rz_scale[:, None, None], precond_radial_alpha)
                    if frzl.frss is not None
                    else None
                )
                fzsc = _apply_radial_tridi(frzl.fzsc * rz_scale[:, None, None], precond_radial_alpha)
                fzcs = (
                    _apply_radial_tridi(frzl.fzcs * rz_scale[:, None, None], precond_radial_alpha)
                    if frzl.fzcs is not None
                    else None
                )
                flsc = _apply_radial_tridi(frzl.flsc * l_scale[:, None, None], precond_lambda_alpha)
                flcs = (
                    _apply_radial_tridi(frzl.flcs * l_scale[:, None, None], precond_lambda_alpha)
                    if frzl.flcs is not None
                    else None
                )
                frsc = (
                    _apply_radial_tridi(frzl.frsc * rz_scale[:, None, None], precond_radial_alpha)
                    if getattr(frzl, "frsc", None) is not None
                    else jnp.zeros_like(frcc)
                )
                frcs = (
                    _apply_radial_tridi(frzl.frcs * rz_scale[:, None, None], precond_radial_alpha)
                    if getattr(frzl, "frcs", None) is not None
                    else jnp.zeros_like(frcc)
                )
                fzcc = (
                    _apply_radial_tridi(frzl.fzcc * rz_scale[:, None, None], precond_radial_alpha)
                    if getattr(frzl, "fzcc", None) is not None
                    else jnp.zeros_like(fzsc)
                )
                fzss = (
                    _apply_radial_tridi(frzl.fzss * rz_scale[:, None, None], precond_radial_alpha)
                    if getattr(frzl, "fzss", None) is not None
                    else jnp.zeros_like(fzsc)
                )
                flcc = (
                    _apply_radial_tridi(frzl.flcc * l_scale[:, None, None], precond_lambda_alpha)
                    if getattr(frzl, "flcc", None) is not None
                    else jnp.zeros_like(flsc)
                )
                flss = (
                    _apply_radial_tridi(frzl.flss * l_scale[:, None, None], precond_lambda_alpha)
                    if getattr(frzl, "flss", None) is not None
                    else jnp.zeros_like(flsc)
                )
                if timing_detail_enabled and t_precond_apply_start is not None:
                    try:
                        if has_jax():
                            jax.block_until_ready(flsc)
                    except Exception:
                        pass
                    timing_stats["precond_apply"] += time.perf_counter() - float(t_precond_apply_start)

            frzl_pre = TomnspsRZL(
                frcc=frcc,
                frss=frss,
                fzsc=fzsc,
                fzcs=fzcs,
                flsc=flsc,
                flcs=flcs,
                frsc=frsc,
                frcs=frcs,
                fzcc=fzcc,
                fzss=fzss,
                flcc=flcc,
                flss=flss,
            )
            if frzl_lam_pre is not None:
                _maybe_dump_lam_gcl(
                    frzl_pre=frzl_lam_pre,
                    frzl_post=frzl_pre,
                    static=static,
                    iter_idx=int(iter2),
                    delta_s=delta_s,
                )
            _maybe_dump_gc(frzl=frzl_pre, static=static, iter_idx=int(iter2), label="precond")

            # Mode-diagonal preconditioning in (m, n>=0) storage.
            t_precond_mode_start = time.perf_counter() if timing_detail_enabled else None
            if preconditioner_outputs_scaled:
                pass
            elif host_update_assembly:
                # NumPy path: avoids 36 JAX dispatches (expand_dims + broadcast + mul per array).
                # _zeros_coeff_np replaces np.zeros_like (pre-allocated, avoids 6+ allocs/iter).
                (frcc_u, frss_u, fzsc_u, fzcs_u, flsc_u, flcs_u, frsc_u, frcs_u, fzcc_u, fzss_u, flcc_u, flss_u) = (
                    _mode_weight_force_blocks_np(
                        _ForceBlocks(frcc, frss, fzsc, fzcs, flsc, flcs, frsc, frcs, fzcc, fzss, flcc, flss),
                        w_mode_mn=w_mode_mn_np,
                        zeros_coeff=_zeros_coeff_np,
                    )
                )
            else:
                frcc_u = frcc * w_mode_mn[None, :, :]
                frss_u = (frss if frss is not None else jnp.zeros_like(frcc_u)) * w_mode_mn[None, :, :]
                fzsc_u = fzsc * w_mode_mn[None, :, :]
                fzcs_u = (fzcs if fzcs is not None else jnp.zeros_like(fzsc_u)) * w_mode_mn[None, :, :]
                flsc_u = flsc * w_mode_mn[None, :, :]
                flcs_u = (flcs if flcs is not None else jnp.zeros_like(flsc_u)) * w_mode_mn[None, :, :]
                frsc_u = frsc * w_mode_mn[None, :, :]
                frcs_u = frcs * w_mode_mn[None, :, :]
                fzcc_u = fzcc * w_mode_mn[None, :, :]
                fzss_u = fzss * w_mode_mn[None, :, :]
                flcc_u = flcc * w_mode_mn[None, :, :]
                flss_u = flss * w_mode_mn[None, :, :]
            if timing_detail_enabled and t_precond_mode_start is not None:
                try:
                    if has_jax():
                        jax.block_until_ready(flsc_u)
                except Exception:
                    pass
                timing_stats["precond_mode_scale"] += time.perf_counter() - float(t_precond_mode_start)
            if timing_enabled:
                try:
                    if has_jax() and not timing_detail_enabled:
                        jax.block_until_ready(flsc_u)
                except Exception:
                    pass
                timing_stats["preconditioner"] += time.perf_counter() - float(t_precond_start)
            t_iteration_control_start = time.perf_counter() if timing_enabled else None
            t_iteration_control_fsq1_start = time.perf_counter() if timing_enabled else None

            # VMEC's lambda coefficients can be expressed in multiple scaling
            # conventions (e.g. restart vs. `wout` vs. internal). Allow parity drivers
            # to apply a constant scale to the lambda residual channel before mapping
            # it into coefficient updates.
            if (lambda_update_scale != 1.0) and (not preconditioner_outputs_scaled):
                flsc_u = flsc_u * lambda_update_scale_j
                flcs_u = flcs_u * lambda_update_scale_j
                flcc_u = flcc_u * lambda_update_scale_j
                flss_u = flss_u * lambda_update_scale_j

            if auto_flip_force and it == 0:
                # Choose force direction by a tiny trial step on the VMEC residual
                # (fsqr+fsqz+fsql), not magnetic energy. Energy monotonicity is not a
                # reliable proxy for VMEC's preconditioned convergence metrics.
                w_curr = float(fsqr_f + fsqz_f + fsql_f)
                # Use a probe step that is large enough to be numerically decisive,
                # but still small relative to typical pseudo-time updates.
                dt_probe = min(1e-2, 0.1 * float(time_step))
                dR_dir = dt_probe * _mn_cos_to_signed_physical(frcc_u, frss_u)
                dZ_dir = dt_probe * _mn_sin_to_signed_physical(fzsc_u, fzcs_u)
                dL_dir = dt_probe * _mn_sin_to_signed_physical_lambda(flsc_u, flcs_u)
                if bool(cfg.lasym):
                    dR_sin_dir = dt_probe * _mn_sin_to_signed_physical(frsc_u, frcs_u)
                    dZ_cos_dir = dt_probe * _mn_cos_to_signed_physical(fzcc_u, fzss_u)
                    dL_cos_dir = dt_probe * _mn_cos_to_signed_physical_lambda(flcc_u, flss_u)
                else:
                    dR_sin_dir = jnp.zeros_like(dR_dir)
                    dZ_cos_dir = jnp.zeros_like(dR_dir)
                    dL_cos_dir = jnp.zeros_like(dR_dir)

                def _trial(sign: float) -> float:
                    st_try = VMECState(
                        layout=state.layout,
                        Rcos=jnp.asarray(state.Rcos) + sign * dR_dir,
                        Rsin=jnp.asarray(state.Rsin) + sign * dR_sin_dir,
                        Zcos=jnp.asarray(state.Zcos) + sign * dZ_cos_dir,
                        Zsin=jnp.asarray(state.Zsin) + sign * dZ_dir,
                        Lcos=jnp.asarray(state.Lcos) + sign * dL_cos_dir,
                        Lsin=jnp.asarray(state.Lsin) + sign * dL_dir,
                    )
                    t_auto_flip_force_start = time.perf_counter() if timing_detail_enabled else None
                    _, _, gcr2_t, gcz2_t, gcl2_t, _, _, norms_t = _compute_forces_iter(
                        st_try,
                        include_edge=True,
                        zero_m1=zero_m1,
                        freeb_bsqvac_half=_freeb_bsqvac_half_for_trial_state(st_try),
                        constraint_precond_diag=constraint_precond_diag,
                        constraint_tcon=constraint_tcon_override,
                        constraint_precond_active=constraint_precond_active,
                        constraint_tcon_active=constraint_tcon_active,
                        iter2=iter2,
                    )
                    _record_compute_force_timing("auto_flip", t_auto_flip_force_start, gcr2_t)
                    fsqr_t, fsqz_t, fsql_t = _fsq_from_norms(
                        norms_t,
                        gcr2_in=gcr2_t,
                        gcz2_in=gcz2_t,
                        gcl2_in=gcl2_t,
                    )
                    return float(np.asarray(fsqr_t + fsqz_t + fsql_t))

                w_pos = _trial(+1.0)
                w_neg = _trial(-1.0)
                if np.isfinite(w_neg) and np.isfinite(w_pos) and (w_neg < w_pos):
                    flip_sign = -1.0
                    if verbose and not (bool(vmec2000_control) and bool(verbose_vmec2000_table)):
                        print(
                            "[solve_fixed_boundary_residual_iter] flipping force sign "
                            f"(w_curr={w_curr:.3e} w_pos={w_pos:.3e} w_neg={w_neg:.3e})"
                        )

            # Damping for the fixed-point update.
            accepted_control_ptau_host: tuple[float, float] | None = None
            if preconditioner_fsq1_ready:
                pass
            elif host_update_assembly:
                # NumPy path: avoids 6+ JAX dispatches for sum-of-squares.
                gcr2_p, gcz2_p, gcl2_p = vmec_gcx2_from_tomnsps_np(
                    frzl=frzl_pre,
                    include_edge=True,
                )
            else:
                gcr2_p, gcz2_p, gcl2_p = vmec_gcx2_from_tomnsps(
                    frzl=frzl_pre,
                    lconm1=bool(getattr(static.cfg, "lconm1", True)),
                    apply_m1_constraints=False,
                    # VMEC residue.f90 calls getfsq(..., medge=m1) for fsq*1,
                    # i.e. it includes the edge row in preconditioned R/Z norms.
                    include_edge=True,
                    apply_scalxc=False,
                    s=s,
                )
            if host_update_assembly:
                # Fast NumPy path: use cached Python-float fnorm1 directly — no JAX dispatch.
                if (
                    bool(vmec2000_control)
                    and bool(vmec2000_cache_valid)
                    and (not bool(need_bcovar_update))
                    and (cache_rz_norm is not None)
                    and (cache_f_norm1 is not None)
                ):
                    _f_norm1_np = float(cache_f_norm1)
                    rz_norm = cache_rz_norm  # Python float (for history list)
                else:
                    _rz_norm_val = _rz_norm_np(state)
                    _f_norm1_np = (1.0 / _rz_norm_val) if _rz_norm_val != 0.0 else float("inf")
                    rz_norm = _rz_norm_val
                f_norm1 = _f_norm1_np  # alias for history list (Python float)
                _finite = np.isfinite(_f_norm1_np)
                fsqr1 = float(gcr2_p) * _f_norm1_np if _finite else 0.0
                fsqz1 = float(gcz2_p) * _f_norm1_np if _finite else 0.0
                if bool(vmec2000_control):
                    # VMEC2000 `residue.f90`: fsql1 = hs * SUM( (faclam*gcl)**2 ) over all js.
                    _gcl2_full = _lambda_preconditioned_full_norm(frzl_pre, use_jax=False)
                    fsql1 = _gcl2_full * delta_s
                else:
                    fsql1 = float(gcl2_p) * delta_s
                # Safe values: NaN/Inf → 0 (same semantics as jnp.where below).
                fsqr1_safe = _finite_float_or_zero(fsqr1)
                fsqz1_safe = _finite_float_or_zero(fsqz1)
                fsql1_safe = _finite_float_or_zero(fsql1)
                fsq1 = fsqr1_safe + fsqz1_safe + fsql1_safe
                # host_update_assembly: keep as Python floats — downstream code (history
                # lists, _precond_diag_floats) handles both float and JAX scalar.
                fsqr1 = fsqr1_safe
                fsqz1 = fsqz1_safe
                fsql1 = fsql1_safe
            else:
                # JAX path: set rz_norm and f_norm1 from cache or recompute.
                if (
                    bool(vmec2000_control)
                    and bool(vmec2000_cache_valid)
                    and (not bool(need_bcovar_update))
                    and (cache_rz_norm is not None)
                    and (cache_f_norm1 is not None)
                ):
                    rz_norm = jnp.asarray(cache_rz_norm)
                    f_norm1 = jnp.asarray(cache_f_norm1)
                else:
                    rz_norm = _rz_norm(state)
                    f_norm1 = jnp.where(rz_norm != 0.0, 1.0 / rz_norm, jnp.asarray(float("inf"), dtype=rz_norm.dtype))
                # Avoid inf*0 -> NaN in late-converged iterations when rz_norm=0 and
                # gcx2 terms are exactly zero. VMEC treats these channels as zero.
                finite_fnorm1 = jnp.isfinite(f_norm1)
                fsqr1 = jnp.where(finite_fnorm1, gcr2_p * f_norm1, jnp.asarray(0.0, dtype=jnp.asarray(gcr2_p).dtype))
                fsqz1 = jnp.where(finite_fnorm1, gcz2_p * f_norm1, jnp.asarray(0.0, dtype=jnp.asarray(gcz2_p).dtype))
                if bool(vmec2000_control):
                    # VMEC2000 `residue.f90`: fsql1 = hs * SUM( (faclam*gcl)**2 ) over all js.
                    gcl2_full = _lambda_preconditioned_full_norm(frzl_pre, use_jax=True)
                    fsql1 = gcl2_full * delta_s
                else:
                    fsql1 = gcl2_p * delta_s
            if _env_dump_lam not in ("", "0") and frzl_lam_pre is None:
                gcr2_raw, gcz2_raw, gcl2_raw = vmec_gcx2_from_tomnsps(
                    frzl=frzl,
                    lconm1=bool(getattr(static.cfg, "lconm1", True)),
                    apply_m1_constraints=False,
                    include_edge=True,
                    apply_scalxc=False,
                    s=s,
                )
                fsql1_pre = gcl2_raw * delta_s
                _maybe_dump_lam_fsql1(
                    fsql1_pre=fsql1_pre,
                    fsql1_post=fsql1,
                    static=static,
                    iter_idx=int(iter2),
                )
            if not host_update_assembly:
                # Extremely small late-iteration channels can occasionally surface
                # as NaN/Inf through mixed 0*Inf paths in XLA. VMEC treats these
                # as effectively zero for the preconditioned residual diagnostics.
                if not preconditioner_fsq1_ready:
                    fsqr1_safe = jnp.where(
                        jnp.isfinite(fsqr1),
                        fsqr1,
                        jnp.asarray(0.0, dtype=jnp.asarray(fsqr1).dtype),
                    )
                    fsqz1_safe = jnp.where(
                        jnp.isfinite(fsqz1),
                        fsqz1,
                        jnp.asarray(0.0, dtype=jnp.asarray(fsqz1).dtype),
                    )
                    fsql1_safe = jnp.where(
                        jnp.isfinite(fsql1),
                        fsql1,
                        jnp.asarray(0.0, dtype=jnp.asarray(fsql1).dtype),
                    )
                if preconditioner_fsq1_ready:
                    fsq1_j = fsq1_safe
                else:
                    fsq1_j = fsqr1_safe + fsqz1_safe + fsql1_safe
                use_control_payload = (
                    (not bool(converged_physical))
                    and (bool(reference_mode) or bool(vmec2000_control))
                    and (not bool(badjac_use_state))
                    and (not bool(dump_ptau_state))
                    and os.getenv("VMEC_JAX_DUMP_PTAU", "") in ("", "0")
                    and jax.default_backend() != "cpu"
                )
                control_payload_used = False
                if accepted_control_ptau_payload is not None:
                    try:
                        fsq1_payload, ptau_min_payload, ptau_max_payload = accepted_control_ptau_payload
                        fsq1, min_tau_ptau_payload, max_tau_ptau_payload = _device_get_floats(
                            fsq1_payload,
                            ptau_min_payload,
                            ptau_max_payload,
                        )
                        accepted_control_ptau_host = (min_tau_ptau_payload, max_tau_ptau_payload)
                        control_payload_used = True
                    except Exception:
                        control_payload_used = False
                if (not control_payload_used) and use_control_payload:
                    ptau_arrays = _scan_math_kernel_arrays_from_k(k)
                    payload_fn = _accepted_control_payload_jit()
                    if ptau_arrays is not None and payload_fn is not None:
                        try:
                            fsq1_payload, ptau_min_payload, ptau_max_payload = payload_fn(
                                fsq1_j,
                                *ptau_arrays,
                                _ptau_pshalf_jax,
                                _ptau_ohs_jax,
                            )
                            fsq1, min_tau_ptau_payload, max_tau_ptau_payload = _device_get_floats(
                                fsq1_payload,
                                ptau_min_payload,
                                ptau_max_payload,
                            )
                            accepted_control_ptau_host = (min_tau_ptau_payload, max_tau_ptau_payload)
                            control_payload_used = True
                        except Exception:
                            control_payload_used = False
                if not control_payload_used:
                    fsq1 = float(jax.device_get(fsq1_j))
            if timing_enabled and t_iteration_control_fsq1_start is not None:
                timing_stats["iteration_control_fsq1"] += time.perf_counter() - float(t_iteration_control_fsq1_start)
            precond_diag_host: tuple[float, float, float] | None = None

            def _precond_diag_floats() -> tuple[float, float, float]:
                nonlocal precond_diag_host
                if precond_diag_host is None:
                    precond_diag_host = _device_get_floats(fsqr1_safe, fsqz1_safe, fsql1_safe)
                return precond_diag_host

            if track_history:
                rz_norm_history.append(rz_norm)
                f_norm1_history.append(f_norm1)
                gcr2_p_history.append(gcr2_p)
                gcz2_p_history.append(gcz2_p)
                gcl2_p_history.append(gcl2_p)
                fsq1_history.append(fsq1)
                fsqr1_history.append(fsqr1_safe)
                fsqz1_history.append(fsqz1_safe)
                fsql1_history.append(fsql1_safe)

            if converged_physical:
                if track_history:
                    # Keep per-iteration history channels length-aligned with
                    # fsqr/fsqz/fsql when convergence happens before the update
                    # block. VMEC's table still reports DELT on this row.
                    rec = _residual_iter_history_record(
                        step=0.0,
                        dt_eff=0.0,
                        update_rms=0.0,
                        w_curr=fsqr_f + fsqz_f + fsql_f,
                        w_try=float("nan"),
                        w_try_ratio=float("nan"),
                        restart_path="converged",
                        step_status="converged",
                        restart_reason="none",
                        pre_restart_reason="none",
                        time_step=time_step,
                        res0=res0,
                        res1=res1,
                        fsq_prev=fsq_prev,
                        bad_growth_streak=bad_growth_streak,
                        iter1=iter1,
                        iter2=iter2,
                        fsqr=fsqr_f,
                        fsqz=fsqz_f,
                        fsql=fsql_f,
                        free_boundary_enabled=free_boundary_enabled,
                        freeb_ivac=freeb_ivac,
                        freeb_ivacskip=freeb_ivacskip,
                    )
                    _append_residual_iter_history_record(
                        rec,
                        step_history=step_history,
                        dt_eff_history=dt_eff_history,
                        update_rms_history=update_rms_history,
                        w_curr_history=w_curr_history,
                        w_try_history=w_try_history,
                        w_try_ratio_history=w_try_ratio_history,
                        restart_path_history=restart_path_history,
                        step_status_history=step_status_history,
                        restart_reason_history=restart_reason_history,
                        pre_restart_reason_history=pre_restart_reason_history,
                        time_step_history=time_step_history,
                        res0_history=res0_history,
                        res1_history=res1_history,
                        fsq_prev_history=fsq_prev_history,
                        bad_growth_streak_history=bad_growth_streak_history,
                        iter1_history=iter1_history,
                        iter2_history=iter2_history,
                        grad_rms_history=grad_rms_history,
                        free_boundary_enabled=free_boundary_enabled,
                        freeb_ivac_history=freeb_ivac_history,
                        freeb_ivacskip_history=freeb_ivacskip_history,
                        freeb_full_update_history=freeb_full_update_history,
                    )
                if verbose and not (bool(vmec2000_control) and bool(verbose_vmec2000_table)):
                    print(
                        f"[solve_fixed_boundary_residual_iter] converged: "
                        f"fsqr={fsqr_f:.3e} fsqz={fsqz_f:.3e} fsql={fsql_f:.3e} "
                        f"target={float(fsq_total_target) if fsq_total_target is not None else float(ftol):.3e}",
                        flush=True,
                    )
                if timing_enabled and t_iteration_control_start is not None:
                    timing_stats["iteration_control"] += time.perf_counter() - float(t_iteration_control_start)
                    t_iteration_control_start = None
                if bool(vmec2000_control) and bool(verbose_vmec2000_table):
                    fsqr1_f, fsqz1_f, fsql1_f = _precond_diag_floats()
                    _print_vmec2000_iter_row(
                        iter_idx=int(iter2),
                        fsqr=fsqr_f,
                        fsqz=fsqz_f,
                        fsql=fsql_f,
                        fsqr1=fsqr1_f,
                        fsqz1=fsqz1_f,
                        fsql1=fsql1_f,
                        delt0r=float(time_step),
                        r00=float(r00_last),
                        w_mhd=float(w_vmec_last),
                        z00=float(z00_last),
                    )
                converged = True
                break

            # Jacobian sign-change check (VMEC jacobian.f sets irst=2).
            t_iteration_control_badjac_start = time.perf_counter() if timing_enabled else None
            bad_jacobian = False
            if bool(reference_mode) or bool(vmec2000_control):
                min_tau_ptau = max_tau_ptau = None
                bad_jacobian_ptau = None
                if accepted_control_ptau_host is not None:
                    min_tau_ptau, max_tau_ptau = accepted_control_ptau_host
                else:
                    ptau_min, ptau_max = _ptau_minmax_from_k_host(k)
                    if ptau_min is not None and ptau_max is not None:
                        min_tau_ptau, max_tau_ptau = _device_get_floats(ptau_min, ptau_max)
                if min_tau_ptau is not None and max_tau_ptau is not None:
                    if bool(vmec2000_control):
                        tau_tol = _bad_jacobian_tau_tolerance(
                            ptau_tol=ptau_tol,
                            ptau_tol_rel=0.0,
                            tau_scale=0.0,
                        )
                        bad_jacobian_ptau = (min_tau_ptau < -tau_tol) and (max_tau_ptau > tau_tol)
                    else:
                        tau_scale = max(abs(min_tau_ptau), abs(max_tau_ptau))
                        tau_tol = max(1.0e-12, 1.0e-3 * tau_scale)
                        bad_jacobian_ptau = (min_tau_ptau < -tau_tol) and (max_tau_ptau > tau_tol)

                state_probe = _should_probe_bad_jacobian_state(
                    state_probe=bool(badjac_state_probe),
                    initial_state_probe_iters=int(badjac_initial_state_probe_iters),
                    iter_idx=int(iter2),
                )
                need_state_jac = (
                    badjac_use_state
                    or dump_ptau_state
                    or state_probe
                    or (bad_jacobian_ptau is None)
                    or bool(bad_jacobian_ptau)
                )
                if need_state_jac:
                    if host_update_assembly and (not _tree_has_tracer(state)) and (not _tree_has_tracer(s)):
                        from .vmec_numpy_forces import _numpy_module_patch as _hot_numpy_patch

                        with _hot_numpy_patch():
                            jac_state = vmec_half_mesh_jacobian_from_state(
                                state=state,
                                modes=static.modes,
                                trig=trig,
                                s=s,
                                lconm1=bool(getattr(static.cfg, "lconm1", True)),
                                lthreed=bool(getattr(static.cfg, "lthreed", True)),
                                mask_even=getattr(static, "m_is_even", None),
                                mask_odd=getattr(static, "m_is_odd", None),
                            )
                        tau = np.asarray(jac_state.tau)
                        if int(tau.size) > 0:
                            tau_use = tau[1:] if int(tau.shape[0]) > 1 else tau
                            min_tau_state = float(np.min(tau_use))
                            max_tau_state = float(np.max(tau_use))
                        else:
                            min_tau_state = float("nan")
                            max_tau_state = float("nan")
                    else:
                        jac_state = vmec_half_mesh_jacobian_from_state(
                            state=state,
                            modes=static.modes,
                            trig=trig,
                            s=s,
                            lconm1=bool(getattr(static.cfg, "lconm1", True)),
                            lthreed=bool(getattr(static.cfg, "lthreed", True)),
                            mask_even=getattr(static, "m_is_even", None),
                            mask_odd=getattr(static, "m_is_odd", None),
                        )
                        tau = jnp.asarray(jac_state.tau)
                        if int(tau.size) > 0:
                            tau_use = tau[1:] if int(tau.shape[0]) > 1 else tau
                            tau_min = jnp.min(tau_use)
                            tau_max = jnp.max(tau_use)
                            min_tau_state, max_tau_state = _device_get_floats(tau_min, tau_max)
                        else:
                            min_tau_state = float("nan")
                            max_tau_state = float("nan")
                    if np.isfinite(min_tau_state) and np.isfinite(max_tau_state):
                        if bool(vmec2000_control):
                            tau_tol = _bad_jacobian_tau_tolerance(
                                ptau_tol=ptau_tol,
                                ptau_tol_rel=0.0,
                                tau_scale=0.0,
                            )
                            bad_jacobian_state = (min_tau_state < -tau_tol) and (max_tau_state > tau_tol)
                        else:
                            tau_scale = max(abs(min_tau_state), abs(max_tau_state))
                            tau_tol = max(1.0e-12, 1.0e-3 * tau_scale)
                            bad_jacobian_state = (min_tau_state < -tau_tol) and (max_tau_state > tau_tol)
                    else:
                        bad_jacobian_state = False
                else:
                    min_tau_state = float("nan")
                    max_tau_state = float("nan")
                    bad_jacobian_state = False

                if badjac_use_state:
                    bad_jacobian = bad_jacobian_state
                    min_tau = min_tau_state
                    max_tau = max_tau_state
                else:
                    bad_jacobian = bool(bad_jacobian_ptau) if bad_jacobian_ptau is not None else False
                    min_tau = min_tau_ptau if min_tau_ptau is not None else float("nan")
                    max_tau = max_tau_ptau if max_tau_ptau is not None else float("nan")

                _maybe_dump_ptau(
                    iter_idx=int(iter2),
                    ptau_min=float(min_tau_ptau if min_tau_ptau is not None else float("nan")),
                    ptau_max=float(max_tau_ptau if max_tau_ptau is not None else float("nan")),
                    tau_min_state=min_tau_state if np.isfinite(min_tau_state) else None,
                    tau_max_state=max_tau_state if np.isfinite(max_tau_state) else None,
                    badjac_ptau=bad_jacobian_ptau,
                    badjac_state=bad_jacobian_state,
                    badjac_used=bool(bad_jacobian),
                    mode=badjac_mode,
                    label="iter",
                )

                if np.isfinite(min_tau) and np.isfinite(max_tau):
                    if track_history:
                        min_tau_history.append(min_tau)
                        max_tau_history.append(max_tau)
                        bad_jacobian_history.append(int(bool(bad_jacobian)))
                    if bad_jacobian and _env_dump_badjac not in ("", "0"):
                        dump_dir = _env_dump_dir
                        if dump_dir:
                            try:
                                path = Path(dump_dir) / "bad_jacobian.log"
                                with path.open("a", encoding="utf-8") as f:
                                    f.write(f"iter={iter2} min_tau={min_tau:.6e} max_tau={max_tau:.6e}\n")
                            except Exception:
                                pass
                else:
                    if track_history:
                        min_tau_history.append(float("nan"))
                        max_tau_history.append(float("nan"))
                        bad_jacobian_history.append(0)
            else:
                if track_history:
                    min_tau_history.append(float("nan"))
                    max_tau_history.append(float("nan"))
                    bad_jacobian_history.append(0)

            # VMEC eqsolve: after the first evolve step, if the Jacobian is bad
            # and ijacob==0, retry with an improved axis guess.
            if bool(vmec2000_control) and (not axis_reset_done) and bool(lmove_axis) and (iter2 == 1):
                fsq_curr = fsqr_f + fsqz_f + fsql_f
                huge_initial_forces = (not np.isfinite(fsq_curr)) or (fsq_curr > 1.0e2)
                force_axis_reset_init = bool(force_axis_reset) or (
                    bool(getattr(cfg, "lthreed", True)) and axis_reset_always_3d
                )
                if (not force_axis_reset_init) and axis_reset_fsq_min > 0.0:
                    if np.isfinite(fsq_curr) and (fsq_curr < axis_reset_fsq_min):
                        bad_jacobian = False
                        huge_initial_forces = False
                if bad_jacobian or huge_initial_forces or force_axis_reset_init:
                    if verbose and bool(vmec2000_control) and bool(verbose_vmec2000_table):
                        if bad_jacobian or force_axis_reset_init:
                            print(" INITIAL JACOBIAN CHANGED SIGN!", flush=True)
                        print(" TRYING TO IMPROVE INITIAL MAGNETIC AXIS GUESS", flush=True)
                    state = _reset_axis_from_boundary(state, k_guess=k, full_reset=False, refine_axis_guess=False)
                    if verbose and bool(vmec2000_control) and bool(verbose_vmec2000_table):
                        if axis_reset_coeffs is not None:
                            raxis_cc, _raxis_cs, _zaxis_cc, zaxis_cs = axis_reset_coeffs
                            _print_axis_guess(raxis_cc, zaxis_cs)
                    state_checkpoint = state
                    vRcc, vRss, vZsc, vZcs, vLsc, vLcs = _zero_velocity_blocks_like(
                        vRcc, vRss, vZsc, vZcs, vLsc, vLcs
                    )
                    time_step = float(time_step)
                    ijacob = 1
                    axis_reset_done = True
                    iter1 = iter2
                    freeb_controls_cached = None
                    bad_growth_streak = 0
                    inv_tau = [0.15 / time_step] * k_ndamp
                    vmec2000_cache_valid = False
                    cache_precond_diag = None
                    cache_tcon = None
                    cache_norms = None
                    cache_rz_scale = None
                    cache_l_scale = None
                    cache_rz_norm = None
                    cache_f_norm1 = None
                    cache_prec_rz_mats = None
                    cache_prec_rz_jmax = None
                    cache_prec_lam_prec = None
                    cache_prec_faclam = None
                    cache_prec_lam_debug = None
                    _pop_iteration_histories()
                    prev_rz_fsq = prev_rz_fsq_before
                    # VMEC restarts the iteration after axis reset without
                    # advancing the iteration counter. Emulate that by
                    # repeating iter2==1 on the next loop pass.
                    if iter2 == 1:
                        iter_offset -= 1
                    if timing_enabled and t_iteration_control_badjac_start is not None:
                        timing_stats["iteration_control_badjac"] += time.perf_counter() - float(
                            t_iteration_control_badjac_start
                        )
                    continue
            if timing_enabled and t_iteration_control_badjac_start is not None:
                timing_stats["iteration_control_badjac"] += time.perf_counter() - float(
                    t_iteration_control_badjac_start
                )

            # VMEC-style time-step control: VMEC2000's `TimeStepControl` + `restart_iter`.
            t_iteration_control_vmec_time_start = time.perf_counter() if timing_enabled else None
            if bool(vmec2000_control) and (not skip_time_control):
                # VMEC's TimeStepControl uses the *previous* preconditioned
                # residual (fsq) which is updated at the end of evolve.f.
                tc = _vmec2000_time_control_decision(
                    iter2=int(iter2),
                    iter1=int(iter1),
                    fsq_prev=float(fsq_prev),
                    fsq0_curr=float(fsq0_curr),
                    fsq0_prev=float(fsq0_prev),
                    res0=float(res0),
                    res1=float(res1),
                    bad_jacobian=bool(bad_jacobian),
                    vmec2000_fact=float(vmec2000_fact),
                )
                fsq = tc.fsq
                fsq0 = tc.fsq0
                res0 = tc.res0
                res1 = tc.res1
                irst_tc = tc.irst
                irst_trace = tc.trace_irst
                if tc.initialized:
                    state_checkpoint = state
                    _dump_time_control_trace(
                        stage="init",
                        iter2=int(iter2),
                        iter1=int(iter1),
                        fsq=float(fsq),
                        fsq0=float(fsq0),
                        res0=float(res0),
                        res1=float(res1),
                        time_step=float(time_step),
                        irst=int(irst_trace),
                    )
                    _maybe_dump_checkpoint(
                        iter_idx=int(iter2), fsq=float(fsq), fsq0=float(fsq0), res0=float(res0), res1=float(res1)
                    )
                _dump_time_control_trace(
                    stage="pre",
                    iter2=int(iter2),
                    iter1=int(iter1),
                    fsq=float(fsq),
                    fsq0=float(fsq0),
                    res0=float(res0),
                    res1=float(res1),
                    time_step=float(time_step),
                    irst=int(irst_trace),
                )
                if tc.store_checkpoint:
                    _dump_time_control_trace(
                        stage="checkpoint",
                        iter2=int(iter2),
                        iter1=int(iter1),
                        fsq=float(fsq),
                        fsq0=float(fsq0),
                        res0=float(res0),
                        res1=float(res1),
                        time_step=float(time_step),
                        irst=int(irst_trace),
                    )
                    state_checkpoint = state
                    _maybe_dump_checkpoint(
                        iter_idx=int(iter2), fsq=float(fsq), fsq0=float(fsq0), res0=float(res0), res1=float(res1)
                    )
                if tc.restart:
                    _maybe_dump_time_control(
                        iter_idx=int(iter2),
                        fsq=float(fsq),
                        fsq0=float(fsq0),
                        res0=float(res0),
                        res1=float(res1),
                        time_step=float(time_step),
                    )
                    pre_restart_reason = tc.pre_restart_reason
                    state = state_checkpoint
                    (
                        vRcc,
                        vRss,
                        vZsc,
                        vZcs,
                        vLsc,
                        vLcs,
                        vRsc,
                        vRcs,
                        vZcc,
                        vZss,
                        vLcc,
                        vLss,
                    ) = _zero_velocity_blocks_like(
                        vRcc, vRss, vZsc, vZcs, vLsc, vLcs, vRsc, vRcs, vZcc, vZss, vLcc, vLss
                    )
                    iter1_prev = int(iter1)
                    time_step_prev = float(time_step)
                    _dump_time_control_trace(
                        stage="restart",
                        iter2=int(iter2),
                        iter1=iter1_prev,
                        fsq=float(fsq),
                        fsq0=float(fsq0),
                        res0=float(res0),
                        res1=float(res1),
                        time_step=time_step_prev,
                        irst=int(irst_tc),
                    )
                    # VMEC2000 `restart_iter`: irst=2 (bad-jac) -> dt*0.9,
                    # irst=3 (time-control) -> dt/1.03.
                    if irst_tc == 2:
                        time_step = max(restart_badjac_factor * time_step, 1e-12)
                        ijacob += 1
                        step_status = "restart_bad_jacobian"
                        restart_reason = "bad_jacobian"
                    else:
                        time_step = max(time_step / restart_badprog_factor, 1e-12)
                        step_status = "restart_time_control"
                        restart_reason = "time_control"
                    bad_resets += 1
                    iter1 = iter2
                    freeb_controls_cached = None
                    bad_growth_streak = 0
                    fsq_prev = fsq_prev_before
                    fsq0_prev = fsq0_prev_before
                    inv_tau = [0.15 / time_step] * k_ndamp
                    vmec2000_cache_valid = False
                    cache_precond_diag = None
                    cache_tcon = None
                    cache_norms = None
                    cache_rz_scale = None
                    cache_l_scale = None
                    cache_rz_norm = None
                    cache_f_norm1 = None
                    cache_prec_rz_mats = None
                    cache_prec_rz_jmax = None
                    cache_prec_lam_prec = None
                    cache_prec_faclam = None
                    cache_prec_lam_debug = None
                    force_bcovar_update = True
                    if track_history:
                        rec = _residual_iter_history_record(
                            step=0.0,
                            dt_eff=0.0,
                            update_rms=0.0,
                            w_curr=fsqr_f + fsqz_f + fsql_f,
                            w_try=float("nan"),
                            w_try_ratio=float("nan"),
                            restart_path="vmec2000_bad_jacobian" if irst_tc == 2 else "vmec2000_time_control",
                            step_status=step_status,
                            restart_reason=restart_reason,
                            pre_restart_reason=pre_restart_reason,
                            time_step=time_step,
                            res0=res0,
                            res1=res1,
                            fsq_prev=fsq_prev,
                            bad_growth_streak=bad_growth_streak,
                            iter1=iter1,
                            iter2=iter2,
                            fsqr=fsqr_f,
                            fsqz=fsqz_f,
                            fsql=fsql_f,
                            free_boundary_enabled=free_boundary_enabled,
                            freeb_ivac=freeb_ivac,
                            freeb_ivacskip=freeb_ivacskip,
                        )
                        _append_residual_iter_history_record(
                            rec,
                            step_history=step_history,
                            dt_eff_history=dt_eff_history,
                            update_rms_history=update_rms_history,
                            w_curr_history=w_curr_history,
                            w_try_history=w_try_history,
                            w_try_ratio_history=w_try_ratio_history,
                            restart_path_history=restart_path_history,
                            step_status_history=step_status_history,
                            restart_reason_history=restart_reason_history,
                            pre_restart_reason_history=pre_restart_reason_history,
                            time_step_history=time_step_history,
                            res0_history=res0_history,
                            res1_history=res1_history,
                            fsq_prev_history=fsq_prev_history,
                            bad_growth_streak_history=bad_growth_streak_history,
                            iter1_history=iter1_history,
                            iter2_history=iter2_history,
                            grad_rms_history=grad_rms_history,
                            free_boundary_enabled=free_boundary_enabled,
                            freeb_ivac_history=freeb_ivac_history,
                            freeb_ivacskip_history=freeb_ivacskip_history,
                            freeb_full_update_history=freeb_full_update_history,
                        )
                    _pop_iteration_histories()
                    prev_rz_fsq = prev_rz_fsq_before
                    skip_time_control = True
                    if timing_enabled and t_iteration_control_vmec_time_start is not None:
                        timing_stats["iteration_control_vmec_time"] += time.perf_counter() - float(
                            t_iteration_control_vmec_time_start
                        )
                    continue
            if timing_enabled and t_iteration_control_vmec_time_start is not None:
                timing_stats["iteration_control_vmec_time"] += time.perf_counter() - float(
                    t_iteration_control_vmec_time_start
                )

            # --- time-step control trackers + optional restart triggers ---
            t_iteration_control_restart_start = time.perf_counter() if timing_enabled else None
            restart_decision = _host_restart_decision(
                iter2=int(iter2),
                iter1=int(iter1),
                fsqr=fsqr_f,
                fsqz=fsqz_f,
                fsql=fsql_f,
                fsq1=fsq1,
                fsq_prev=fsq_prev,
                res0=res0,
                bad_growth_streak=bad_growth_streak,
                pre_restart_reason=pre_restart_reason,
                reference_mode=reference_mode,
                vmec2000_control=vmec2000_control,
                bad_jacobian=bad_jacobian,
                stage_prev_fsq=stage_prev_fsq,
                stage_transition_factor=stage_transition_factor,
                lmove_axis=lmove_axis,
                vmecpp_restart=vmecpp_restart,
                k_preconditioner_update_interval=k_preconditioner_update_interval,
            )
            fsq = restart_decision.fsq
            res0 = restart_decision.res0
            bad_growth_streak = restart_decision.bad_growth_streak
            pre_restart_reason = restart_decision.pre_restart_reason
            huge_initial_forces = restart_decision.huge_initial_forces

            # Store a "good" checkpoint once residual has improved for many
            # iterations since the last restart marker.
            if restart_decision.store_checkpoint:
                state_checkpoint = state

            if use_restart_triggers and pre_restart_reason != "none":
                state_before_restart = state
                vRcc_before = vRcc
                vRss_before = vRss
                vZsc_before = vZsc
                vZcs_before = vZcs
                vLsc_before = vLsc
                vLcs_before = vLcs
                vRsc_before = vRsc
                vRcs_before = vRcs
                vZcc_before = vZcc
                vZss_before = vZss
                vLcc_before = vLcc
                vLss_before = vLss
                state = state_checkpoint
                (
                    vRcc,
                    vRss,
                    vZsc,
                    vZcs,
                    vLsc,
                    vLcs,
                    vRsc,
                    vRcs,
                    vZcc,
                    vZss,
                    vLcc,
                    vLss,
                ) = _zero_velocity_blocks_like(vRcc, vRss, vZsc, vZcs, vLsc, vLcs, vRsc, vRcs, vZcc, vZss, vLcc, vLss)
                if pre_restart_reason == "bad_jacobian":
                    time_step = max(restart_badjac_factor * time_step, 1e-12)
                    ijacob += 1
                    step_status = "restart_bad_jacobian"
                elif pre_restart_reason == "stage_transition":
                    time_step = max(time_step * stage_transition_scale, 1e-12)
                    step_status = "restart_stage_transition"
                else:
                    time_step = max(time_step / restart_badprog_factor, 1e-12)
                    step_status = "restart_bad_progress"
                if bool(huge_initial_forces) and (pre_restart_reason == "bad_jacobian"):
                    huge_force_restart_count += 1
                else:
                    huge_force_restart_count = 0
                if ijacob in (25, 50):
                    scale = 0.98 if ijacob < 50 else 0.96
                    time_step = max(scale * float(step_size), 1e-12)
                time_step_iter = float(time_step)
                bad_resets += 1
                iter1 = iter2
                freeb_controls_cached = None
                bad_growth_streak = 0
                fsq_prev = fsq_prev_before
                fsq0_prev = fsq0_prev_before
                inv_tau = [0.15 / time_step] * k_ndamp
                if not bool(vmec2000_control):
                    vmec2000_cache_valid = False
                    cache_precond_diag = None
                    cache_tcon = None
                    cache_norms = None
                    cache_rz_scale = None
                    cache_l_scale = None
                    cache_rz_norm = None
                    cache_f_norm1 = None
                    cache_prec_rz_mats = None
                    cache_prec_rz_jmax = None
                    cache_prec_lam_prec = None
                    cache_prec_faclam = None
                    cache_prec_lam_debug = None
                else:
                    vmec2000_cache_valid = False
                    cache_precond_diag = None
                    cache_tcon = None
                    cache_norms = None
                    cache_rz_scale = None
                    cache_l_scale = None
                    cache_rz_norm = None
                    cache_f_norm1 = None
                    cache_prec_rz_mats = None
                    cache_prec_rz_jmax = None
                    cache_prec_lam_prec = None
                    cache_prec_faclam = None
                    cache_prec_lam_debug = None
                    force_bcovar_update = True
                if track_history:
                    rec = _residual_iter_history_record(
                        step=0.0,
                        dt_eff=0.0,
                        update_rms=0.0,
                        w_curr=fsqr_f + fsqz_f + fsql_f,
                        w_try=float("nan"),
                        w_try_ratio=float("nan"),
                        restart_path="pre_restart_trigger",
                        step_status=step_status,
                        restart_reason=pre_restart_reason,
                        pre_restart_reason=pre_restart_reason,
                        time_step=time_step_iter,
                        res0=res0,
                        res1=res1,
                        fsq_prev=fsq_prev,
                        bad_growth_streak=bad_growth_streak,
                        iter1=iter1,
                        iter2=iter2,
                        fsqr=fsqr_f,
                        fsqz=fsqz_f,
                        fsql=fsql_f,
                        free_boundary_enabled=free_boundary_enabled,
                        freeb_ivac=freeb_ivac,
                        freeb_ivacskip=freeb_ivacskip,
                    )
                    _append_residual_iter_history_record(
                        rec,
                        step_history=step_history,
                        dt_eff_history=dt_eff_history,
                        update_rms_history=update_rms_history,
                        w_curr_history=w_curr_history,
                        w_try_history=w_try_history,
                        w_try_ratio_history=w_try_ratio_history,
                        restart_path_history=restart_path_history,
                        step_status_history=step_status_history,
                        restart_reason_history=restart_reason_history,
                        pre_restart_reason_history=pre_restart_reason_history,
                        time_step_history=time_step_history,
                        res0_history=res0_history,
                        res1_history=res1_history,
                        fsq_prev_history=fsq_prev_history,
                        bad_growth_streak_history=bad_growth_streak_history,
                        iter1_history=iter1_history,
                        iter2_history=iter2_history,
                        grad_rms_history=grad_rms_history,
                        free_boundary_enabled=free_boundary_enabled,
                        freeb_ivac_history=freeb_ivac_history,
                        freeb_ivacskip_history=freeb_ivacskip_history,
                        freeb_full_update_history=freeb_full_update_history,
                    )
                if verbose:
                    if bool(vmec2000_control) and bool(verbose_vmec2000_table):
                        # VMEC does not print rejected restart steps.
                        pass
                    else:
                        fsqr1_f, fsqz1_f, fsql1_f = _precond_diag_floats()
                        print(
                            f"[solve_fixed_boundary_residual_iter] iter={it:03d} "
                            f"dt_eff=0.000e+00 update_rms=0.000e+00 "
                            f"fsqr1={fsqr1_f:.3e} fsqz1={fsqz1_f:.3e} fsql1={fsql1_f:.3e} "
                            f"step_status={step_status}",
                            flush=True,
                        )
                _maybe_dump_xc(
                    state=state_before_restart,
                    vRcc=vRcc_before,
                    vRss=vRss_before,
                    vZsc=vZsc_before,
                    vZcs=vZcs_before,
                    vLsc=vLsc_before,
                    vLcs=vLcs_before,
                    vRsc=vRsc_before,
                    vRcs=vRcs_before,
                    vZcc=vZcc_before,
                    vZss=vZss_before,
                    vLcc=vLcc_before,
                    vLss=vLss_before,
                    static=static,
                    iter_idx=int(iter2),
                )
                _pop_iteration_histories()
                prev_rz_fsq = prev_rz_fsq_before
                skip_time_control = True
                if timing_enabled and t_iteration_control_restart_start is not None:
                    timing_stats["iteration_control_restart"] += time.perf_counter() - float(
                        t_iteration_control_restart_start
                    )
                continue

            if timing_enabled and t_iteration_control_restart_start is not None:
                timing_stats["iteration_control_restart"] += time.perf_counter() - float(
                    t_iteration_control_restart_start
                )
            break
        if profile_started and (profile_start_iter is not None) and (iter2 == profile_start_iter):
            if has_jax():
                try:
                    jax.block_until_ready(state.Rcos)
                    jax.profiler.stop_trace()
                except Exception:
                    pass
            profile_started = False
            profile_active = False
        if converged:
            break
        t_iteration_control_evolve_start = time.perf_counter() if timing_enabled else None
        if iter2 == iter1:
            inv_tau = [0.15 / time_step] * k_ndamp
        else:
            invtau_num = 0.0 if fsq1 == 0.0 else min(abs(np.log(fsq1 / fsq_prev)), 0.15)
            inv_tau = inv_tau[1:] + [invtau_num / time_step]
        fsq_prev = fsq1
        fsq0_prev = fsq0_curr

        otav = float(np.sum(inv_tau)) / float(k_ndamp)
        dtau = time_step * otav / 2.0
        b1 = 1.0 - dtau
        fac = 1.0 / (1.0 + dtau)
        _dump_evolve_trace(
            iter2=int(iter2),
            iter1=int(iter1),
            stage="pre",
            fsq1_val=float(fsq1),
            fsq_prev_val=float(fsq_prev_before),
            time_step_val=float(time_step),
            dtau_val=float(dtau),
            b1_val=float(b1),
            fac_val=float(fac),
            state_val=state,
            vRcc_val=vRcc,
            vRss_val=vRss,
            vZsc_val=vZsc,
            vZcs_val=vZcs,
            vLsc_val=vLsc,
            vLcs_val=vLcs,
            vRsc_val=vRsc,
            vRcs_val=vRcs,
            vZcc_val=vZcc,
            vZss_val=vZss,
            vLcc_val=vLcc,
            vLss_val=vLss,
            frcc_val=frcc_u,
            frss_val=frss_u,
            fzsc_val=fzsc_u,
            fzcs_val=fzcs_u,
            flsc_val=flsc_u,
            flcs_val=flcs_u,
            frsc_val=frsc_u,
            frcs_val=frcs_u,
            fzcc_val=fzcc_u,
            fzss_val=fzss_u,
            flcc_val=flcc_u,
            flss_val=flss_u,
        )

        if timing_enabled and t_iteration_control_evolve_start is not None:
            timing_stats["iteration_control_evolve"] += time.perf_counter() - float(t_iteration_control_evolve_start)
        if timing_enabled and t_iteration_control_start is not None:
            timing_stats["iteration_control"] += time.perf_counter() - float(t_iteration_control_start)
            t_iteration_control_start = None
        t_update_start = time.perf_counter() if timing_enabled else None
        if bool(strict_update):
            # Strict update semantics: one preconditioned momentum update per
            # iteration in (m, n>=0) storage, no line-search accept/reject.
            w_curr = fsqr_f + fsqz_f + fsql_f
            state_backup = state
            t_trace_build_start = time.perf_counter() if timing_enabled and adjoint_trace else None
            if adjoint_trace:
                trace_entry: dict[str, Any] = {
                    "branch": "strict_update",
                    "state_pre": state_backup,
                    "max_update_rms_pre": float(max_update_rms),
                    "max_coeff_delta_rms_pre": float(max_coeff_delta_rms),
                    "divide_by_scalxc_for_update": bool(divide_by_scalxc_for_update),
                    "lambda_update_scale": float(lambda_update_scale),
                    "apply_lforbal": bool(apply_lforbal),
                    "apply_m1_constraints": bool(apply_m1_constraints),
                    "include_edge_residual": bool(include_edge_residual),
                    "vmec2000_control": bool(vmec2000_control),
                    "limit_dt_from_force": bool(limit_dt_from_force),
                    "signgs": int(signgs),
                    "zero_m1": _adjoint_trace_array(zero_m1),
                    "wout_like": wout_like,
                    "trig": trig,
                    "w_mode_mn": _adjoint_trace_array(w_mode_mn),
                    "precond_jmax": int(jmax),
                    "preconditioner_use_precomputed_tridi": bool(preconditioner_use_precomputed_tridi_policy),
                    "preconditioner_use_lax_tridi": bool(preconditioner_use_lax_tridi_policy),
                    "inv_tau_before": _adjoint_trace_array(inv_tau),
                    "fsq_prev_before": float(fsq_prev_before),
                    "reset_inv_tau": bool(iter2 == iter1),
                    "vRcc_before": _adjoint_trace_array(vRcc),
                    "vRss_before": _adjoint_trace_array(vRss),
                    "vZsc_before": _adjoint_trace_array(vZsc),
                    "vZcs_before": _adjoint_trace_array(vZcs),
                    "vLsc_before": _adjoint_trace_array(vLsc),
                    "vLcs_before": _adjoint_trace_array(vLcs),
                    "vRsc_before": _adjoint_trace_array(vRsc),
                    "vRcs_before": _adjoint_trace_array(vRcs),
                    "vZcc_before": _adjoint_trace_array(vZcc),
                    "vZss_before": _adjoint_trace_array(vZss),
                    "vLcc_before": _adjoint_trace_array(vLcc),
                    "vLss_before": _adjoint_trace_array(vLss),
                    "freeb_bsqvac_half": (
                        None
                        if freeb_bsqvac_half_current is None
                        else _adjoint_trace_array(freeb_bsqvac_half_current)
                    ),
                    "freeb_pres_scale": None if freeb_pres_scale is None else float(freeb_pres_scale),
                }
                if adjoint_trace_mode == "full":
                    trace_entry.update(
                        {
                            "lam_prec": np.asarray(lam_prec),
                            "precond_mats": mats,
                            "frzl_frcc": np.asarray(frzl.frcc),
                            "frzl_frss": None if frzl.frss is None else np.asarray(frzl.frss),
                            "frzl_fzsc": np.asarray(frzl.fzsc),
                            "frzl_fzcs": None if frzl.fzcs is None else np.asarray(frzl.fzcs),
                            "frzl_flsc": np.asarray(frzl.flsc),
                            "frzl_flcs": None if frzl.flcs is None else np.asarray(frzl.flcs),
                            "frzl_frsc": None if getattr(frzl, "frsc", None) is None else np.asarray(frzl.frsc),
                            "frzl_frcs": None if getattr(frzl, "frcs", None) is None else np.asarray(frzl.frcs),
                            "frzl_fzcc": None if getattr(frzl, "fzcc", None) is None else np.asarray(frzl.fzcc),
                            "frzl_fzss": None if getattr(frzl, "fzss", None) is None else np.asarray(frzl.fzss),
                            "frzl_flcc": None if getattr(frzl, "flcc", None) is None else np.asarray(frzl.flcc),
                            "frzl_flss": None if getattr(frzl, "flss", None) is None else np.asarray(frzl.flss),
                            "frzl_rz_frcc": np.asarray(frzl_rz.frcc),
                            "frzl_rz_frss": None if frzl_rz.frss is None else np.asarray(frzl_rz.frss),
                            "frzl_rz_fzsc": np.asarray(frzl_rz.fzsc),
                            "frzl_rz_fzcs": None if frzl_rz.fzcs is None else np.asarray(frzl_rz.fzcs),
                            "frzl_rz_flsc": np.asarray(frzl_rz.flsc),
                            "frzl_rz_flcs": None if frzl_rz.flcs is None else np.asarray(frzl_rz.flcs),
                            "frzl_rz_frsc": None
                            if getattr(frzl_rz, "frsc", None) is None
                            else np.asarray(frzl_rz.frsc),
                            "frzl_rz_frcs": None
                            if getattr(frzl_rz, "frcs", None) is None
                            else np.asarray(frzl_rz.frcs),
                            "frzl_rz_fzcc": None
                            if getattr(frzl_rz, "fzcc", None) is None
                            else np.asarray(frzl_rz.fzcc),
                            "frzl_rz_fzss": None
                            if getattr(frzl_rz, "fzss", None) is None
                            else np.asarray(frzl_rz.fzss),
                            "frzl_rz_flcc": None
                            if getattr(frzl_rz, "flcc", None) is None
                            else np.asarray(frzl_rz.flcc),
                            "frzl_rz_flss": None
                            if getattr(frzl_rz, "flss", None) is None
                            else np.asarray(frzl_rz.flss),
                            "frcc_u": np.asarray(frcc_u),
                            "frss_u": np.asarray(frss_u),
                            "fzsc_u": np.asarray(fzsc_u),
                            "fzcs_u": np.asarray(fzcs_u),
                            "flsc_u": np.asarray(flsc_u),
                            "flcs_u": np.asarray(flcs_u),
                            "frsc_u": np.asarray(frsc_u),
                            "frcs_u": np.asarray(frcs_u),
                            "fzcc_u": np.asarray(fzcc_u),
                            "fzss_u": np.asarray(fzss_u),
                            "flcc_u": np.asarray(flcc_u),
                            "flss_u": np.asarray(flss_u),
                        }
                    )
            if timing_enabled and t_trace_build_start is not None:
                timing_stats["update_trace_build"] += time.perf_counter() - float(t_trace_build_start)
            t_state_update_start = time.perf_counter() if timing_enabled else None
            dt_eff = float(time_step)
            if bool(limit_dt_from_force):
                dt_eff = _safe_dt_from_force(
                    dt_nominal=time_step,
                    frcc=frcc_u,
                    frss=frss_u,
                    fzsc=fzsc_u,
                    fzcs=fzcs_u,
                    flsc=flsc_u,
                    flcs=flcs_u,
                    frsc=frsc_u,
                    frcs=frcs_u,
                    fzcc=fzcc_u,
                    fzss=fzss_u,
                    flcc=flcc_u,
                    flss=flss_u,
                )

            # Momentum semantics: v <- fac*(b1*v + dt*F), x <- x + dt*v.
            # Do not drop the dt factor in the force term; otherwise updates
            # scale like O(dt) instead of O(dt^2) and can immediately blow up.
            force_scale = float(dt_eff)

            need_update_rms = (
                bool(limit_update_rms)
                or bool(track_history)
                or bool(verbose)
                or bool(backtracking)
                or (bool(adjoint_trace) and adjoint_trace_mode == "full")
            )
            need_trial_eval = bool(backtracking) or bool(reference_mode) or bool(use_direct_fallback)
            use_jit_strict_update_step = (
                bool(jit_strict_update_enabled)
                and (not bool(host_update_assembly))
                and (not bool(limit_dt_from_force))
                and (not bool(limit_update_rms))
                and (not bool(need_trial_eval))
                and (not _tree_has_tracer(state))
            )
            if use_jit_strict_update_step:
                step_fn = _strict_update_step_jit(
                    static,
                    limit_update_rms=False,
                    need_update_rms=need_update_rms,
                    divide_by_scalxc_for_update=bool(divide_by_scalxc_for_update),
                    enforce_edge=not bool(free_boundary_enabled),
                )
                step_out = step_fn(
                    state,
                    dt_eff,
                    b1,
                    fac,
                    force_scale,
                    flip_sign,
                    vRcc,
                    vRss,
                    vZsc,
                    vZcs,
                    vLsc,
                    vLcs,
                    vRsc,
                    vRcs,
                    vZcc,
                    vZss,
                    vLcc,
                    vLss,
                    frcc_u,
                    frss_u,
                    fzsc_u,
                    fzcs_u,
                    flsc_u,
                    flcs_u,
                    frsc_u,
                    frcs_u,
                    fzcc_u,
                    fzss_u,
                    flcc_u,
                    flss_u,
                    max_update_rms,
                )
                state_try = step_out["state_post"]
                vRcc = step_out["vRcc_after"]
                vRss = step_out["vRss_after"]
                vZsc = step_out["vZsc_after"]
                vZcs = step_out["vZcs_after"]
                vLsc = step_out["vLsc_after"]
                vLcs = step_out["vLcs_after"]
                vRsc = step_out["vRsc_after"]
                vRcs = step_out["vRcs_after"]
                vZcc = step_out["vZcc_after"]
                vZss = step_out["vZss_after"]
                vLcc = step_out["vLcc_after"]
                vLss = step_out["vLss_after"]
                update_rms_j = (
                    step_out["update_rms_postclip"]
                    if need_update_rms
                    else jnp.asarray(0.0, dtype=jnp.asarray(vRcc).dtype)
                )
                update_rms = None
                update_rms_preclip = None
                scl = 1.0
            elif host_update_assembly:
                host_update = _host_momentum_update_np(
                    velocities=_ResidualVelocityBlocks(
                        vRcc,
                        vRss,
                        vRsc,
                        vRcs,
                        vZsc,
                        vZcs,
                        vZcc,
                        vZss,
                        vLsc,
                        vLcs,
                        vLcc,
                        vLss,
                    ),
                    forces=_ResidualVelocityBlocks(
                        frcc_u,
                        frss_u,
                        frsc_u,
                        frcs_u,
                        fzsc_u,
                        fzcs_u,
                        fzcc_u,
                        fzss_u,
                        flsc_u,
                        flcs_u,
                        flcc_u,
                        flss_u,
                    ),
                    b1=b1,
                    fac=fac,
                    force_scale=force_scale,
                    flip_sign=flip_sign,
                    dt_eff=dt_eff,
                    compute_update_rms=need_update_rms,
                )
                # Unpack as NumPy array views — no JAX conversion here.
                (vRcc, vRss, vRsc, vRcs, vZsc, vZcs, vZcc, vZss, vLsc, vLcs, vLcc, vLss) = (
                    host_update.velocities
                )
                if need_update_rms:
                    update_rms_j = host_update.update_rms
                else:
                    update_rms_j = jnp.asarray(0.0, dtype=jnp.asarray(vRcc).dtype)
            else:
                vRcc = fac * (b1 * vRcc + force_scale * (flip_sign * jnp.asarray(frcc_u)))
                vRss = fac * (b1 * vRss + force_scale * (flip_sign * jnp.asarray(frss_u)))
                vRsc = fac * (b1 * vRsc + force_scale * (flip_sign * jnp.asarray(frsc_u)))
                vRcs = fac * (b1 * vRcs + force_scale * (flip_sign * jnp.asarray(frcs_u)))
                vZsc = fac * (b1 * vZsc + force_scale * (flip_sign * jnp.asarray(fzsc_u)))
                vZcs = fac * (b1 * vZcs + force_scale * (flip_sign * jnp.asarray(fzcs_u)))
                vZcc = fac * (b1 * vZcc + force_scale * (flip_sign * jnp.asarray(fzcc_u)))
                vZss = fac * (b1 * vZss + force_scale * (flip_sign * jnp.asarray(fzss_u)))
                vLsc = fac * (b1 * vLsc + force_scale * (flip_sign * jnp.asarray(flsc_u)))
                vLcs = fac * (b1 * vLcs + force_scale * (flip_sign * jnp.asarray(flcs_u)))
                vLcc = fac * (b1 * vLcc + force_scale * (flip_sign * jnp.asarray(flcc_u)))
                vLss = fac * (b1 * vLss + force_scale * (flip_sign * jnp.asarray(flss_u)))
                if need_update_rms:
                    update_rms_j = jnp.sqrt(
                        jnp.mean(
                            (dt_eff * vRcc) ** 2
                            + (dt_eff * vRss) ** 2
                            + (dt_eff * vRsc) ** 2
                            + (dt_eff * vRcs) ** 2
                            + (dt_eff * vZsc) ** 2
                            + (dt_eff * vZcs) ** 2
                            + (dt_eff * vZcc) ** 2
                            + (dt_eff * vZss) ** 2
                            + (dt_eff * vLsc) ** 2
                            + (dt_eff * vLcs) ** 2
                            + (dt_eff * vLcc) ** 2
                            + (dt_eff * vLss) ** 2
                        )
                    )
                else:
                    update_rms_j = jnp.asarray(0.0, dtype=jnp.asarray(vRcc).dtype)

            if not use_jit_strict_update_step:
                update_rms_host: float | None = None

                def _update_rms_float() -> float:
                    nonlocal update_rms_host
                    if update_rms_host is None:
                        update_rms_host = float(np.asarray(update_rms_j))
                    return update_rms_host

                if (
                    bool(limit_update_rms)
                    or bool(backtracking)
                    or (bool(adjoint_trace) and adjoint_trace_mode == "full")
                ):
                    update_rms = _update_rms_float()
                else:
                    update_rms = None
                update_rms_preclip = update_rms
                if bool(limit_update_rms) and np.isfinite(update_rms) and (update_rms > max_update_rms):
                    scl = max_update_rms / max(update_rms, 1e-30)
                    vRcc = vRcc * scl
                    vRss = vRss * scl
                    vRsc = vRsc * scl
                    vRcs = vRcs * scl
                    vZsc = vZsc * scl
                    vZcs = vZcs * scl
                    vZcc = vZcc * scl
                    vZss = vZss * scl
                    vLsc = vLsc * scl
                    vLcs = vLcs * scl
                    vLcc = vLcc * scl
                    vLss = vLss * scl
                    update_rms_j = jnp.sqrt(
                        jnp.mean(
                            (dt_eff * vRcc) ** 2
                            + (dt_eff * vRss) ** 2
                            + (dt_eff * vRsc) ** 2
                            + (dt_eff * vRcs) ** 2
                            + (dt_eff * vZsc) ** 2
                            + (dt_eff * vZcs) ** 2
                            + (dt_eff * vZcc) ** 2
                            + (dt_eff * vZss) ** 2
                            + (dt_eff * vLsc) ** 2
                            + (dt_eff * vLcs) ** 2
                            + (dt_eff * vLcc) ** 2
                            + (dt_eff * vLss) ** 2
                        )
                    )
                    update_rms_host = float(np.asarray(update_rms_j))
                    update_rms = update_rms_host
                else:
                    scl = 1.0

                dR = dt_eff * _mn_cos_to_signed_physical(vRcc, vRss)
                dZ = dt_eff * _mn_sin_to_signed_physical(vZsc, vZcs)
                dL = dt_eff * _mn_sin_to_signed_physical_lambda(vLsc, vLcs)
                if bool(cfg.lasym):
                    dR_sin = dt_eff * _mn_sin_to_signed_physical(vRsc, vRcs)
                    dZ_cos = dt_eff * _mn_cos_to_signed_physical(vZcc, vZss)
                    dL_cos = dt_eff * _mn_cos_to_signed_physical_lambda(vLcc, vLss)
                else:
                    if host_update_assembly:
                        # Use pre-allocated zero arrays (avoid 3 np.zeros_like allocs/iter).
                        dR_sin = _zeros_dR_np
                        dZ_cos = _zeros_dR_np
                        dL_cos = _zeros_dR_np
                    else:
                        dR_sin = jnp.zeros_like(dR)
                        dZ_cos = jnp.zeros_like(dR)
                        dL_cos = jnp.zeros_like(dR)
                if host_update_assembly:
                    # All dR/dZ/dL/dR_sin/dZ_cos/dL_cos are NumPy here;
                    # keep state arrays as NumPy — JAX JIT converts at call site.
                    state_try = VMECState(
                        layout=state.layout,
                        Rcos=np.asarray(state.Rcos) + np.asarray(dR),
                        Rsin=np.asarray(state.Rsin) + np.asarray(dR_sin),
                        Zcos=np.asarray(state.Zcos) + np.asarray(dZ_cos),
                        Zsin=np.asarray(state.Zsin) + np.asarray(dZ),
                        Lcos=np.asarray(state.Lcos) + np.asarray(dL_cos),
                        Lsin=np.asarray(state.Lsin) + np.asarray(dL),
                    )
                else:
                    state_try = VMECState(
                        layout=state.layout,
                        Rcos=jnp.asarray(state.Rcos) + dR,
                        Rsin=jnp.asarray(state.Rsin) + dR_sin,
                        Zcos=jnp.asarray(state.Zcos) + dZ_cos,
                        Zsin=jnp.asarray(state.Zsin) + dZ,
                        Lcos=jnp.asarray(state.Lcos) + dL_cos,
                        Lsin=jnp.asarray(state.Lsin) + dL,
                    )
                if host_update_assembly:
                    state_try = _enforce_fixed_boundary_and_axis_np(
                        state_try,
                        static,
                        edge_Rcos=edge_Rcos,
                        edge_Rsin=edge_Rsin,
                        edge_Zcos=edge_Zcos,
                        edge_Zsin=edge_Zsin,
                        enforce_edge=not bool(free_boundary_enabled),
                        enforce_lambda_axis=True,
                        idx00=idx00,
                        precomputed_axis_mask=_precomputed_axis_mask_np,
                    )
                else:
                    state_try = _enforce_fixed_boundary_and_axis(
                        state_try,
                        static,
                        edge_Rcos=edge_Rcos,
                        edge_Rsin=edge_Rsin,
                        edge_Zcos=edge_Zcos,
                        edge_Zsin=edge_Zsin,
                        enforce_edge=not bool(free_boundary_enabled),
                        enforce_lambda_axis=True,
                        idx00=idx00,
                    )
                state_try = _apply_vmec_lambda_axis_rules(state_try)
            probe_bad_jacobian = False
            if need_trial_eval:
                freeb_bsqvac_half_trial = _freeb_bsqvac_half_for_trial_state(state_try)
                t_trial_force_start = time.perf_counter() if timing_detail_enabled else None
                _, _, gcr2_t, gcz2_t, gcl2_t, _, _, norms_t = _compute_forces_iter(
                    state_try,
                    include_edge=include_edge,
                    zero_m1=zero_m1,
                    freeb_bsqvac_half=freeb_bsqvac_half_trial,
                    constraint_precond_diag=constraint_precond_diag,
                    constraint_tcon=constraint_tcon_override,
                    constraint_precond_active=constraint_precond_active,
                    constraint_tcon_active=constraint_tcon_active,
                    iter2=iter2,
                )
                _record_compute_force_timing("trial", t_trial_force_start, gcr2_t)
                fsqr_t, fsqz_t, fsql_t = _fsq_from_norms(
                    norms_t,
                    gcr2_in=gcr2_t,
                    gcz2_in=gcz2_t,
                    gcl2_in=gcl2_t,
                )
                w_try = float(np.asarray(fsqr_t + fsqz_t + fsql_t))
                w_try_ratio = w_try / max(w_curr, 1e-30) if np.isfinite(w_try) else float("inf")
                if bool(reference_mode) and (float(np.asarray(zero_m1)) > 0.5):
                    _, _, gcr2_probe, gcz2_probe, gcl2_probe, _, _, norms_probe = _compute_forces_iter(
                        state_try,
                        include_edge=include_edge,
                        zero_m1=jnp.asarray(0.0, dtype=zero_m1.dtype),
                        freeb_bsqvac_half=freeb_bsqvac_half_trial,
                        constraint_precond_diag=constraint_precond_diag,
                        constraint_tcon=constraint_tcon_override,
                        constraint_precond_active=constraint_precond_active,
                        constraint_tcon_active=constraint_tcon_active,
                        iter2=iter2,
                    )
                    fsqr_probe, fsqz_probe, fsql_probe = _fsq_from_norms(
                        norms_probe,
                        gcr2_in=gcr2_probe,
                        gcz2_in=gcz2_probe,
                        gcl2_in=gcl2_probe,
                    )
                    w_probe = float(np.asarray(fsqr_probe + fsqz_probe + fsql_probe))
                    if (not np.isfinite(w_probe)) or (w_probe > 1.0e2 * max(w_curr, 1e-30)):
                        probe_bad_jacobian = True
                        w_try = float("inf")
                        w_try_ratio = float("inf")
            else:
                w_try = w_curr
                w_try_ratio = 1.0

            # The reference iteration is typically stable under its restart
            # triggers, but our parity-path preconditioners are still evolving.
            # Add a small,
            # bounded backtracking on the position update (not the force
            # evaluation) to prevent systematic residual growth.
            alpha = 1.0
            accept_ratio = 1.001 if backtracking else float("inf")
            if np.isfinite(w_try) and (w_try > accept_ratio * max(w_curr, 1e-30)):
                for _ in range(8):
                    alpha *= 0.5
                    state_try = VMECState(
                        layout=state.layout,
                        Rcos=jnp.asarray(state.Rcos) + alpha * dR,
                        Rsin=jnp.asarray(state.Rsin) + alpha * dR_sin,
                        Zcos=jnp.asarray(state.Zcos) + alpha * dZ_cos,
                        Zsin=jnp.asarray(state.Zsin) + alpha * dZ,
                        Lcos=jnp.asarray(state.Lcos) + alpha * dL_cos,
                        Lsin=jnp.asarray(state.Lsin) + alpha * dL,
                    )
                    if host_update_assembly:
                        state_try = _enforce_fixed_boundary_and_axis_np(
                            state_try,
                            static,
                            edge_Rcos=edge_Rcos,
                            edge_Rsin=edge_Rsin,
                            edge_Zcos=edge_Zcos,
                            edge_Zsin=edge_Zsin,
                            enforce_edge=not bool(free_boundary_enabled),
                            enforce_lambda_axis=True,
                            idx00=idx00,
                            precomputed_axis_mask=_precomputed_axis_mask_np,
                        )
                    else:
                        state_try = _enforce_fixed_boundary_and_axis(
                            state_try,
                            static,
                            edge_Rcos=edge_Rcos,
                            edge_Rsin=edge_Rsin,
                            edge_Zcos=edge_Zcos,
                            edge_Zsin=edge_Zsin,
                            enforce_edge=not bool(free_boundary_enabled),
                            enforce_lambda_axis=True,
                            idx00=idx00,
                    )
                    state_try = _apply_vmec_lambda_axis_rules(state_try)
                    freeb_bsqvac_half_trial = _freeb_bsqvac_half_for_trial_state(state_try)
                    t_trial_force_start = time.perf_counter() if timing_detail_enabled else None
                    _, _, gcr2_t, gcz2_t, gcl2_t, _, _, norms_t = _compute_forces_iter(
                        state_try,
                        include_edge=include_edge,
                        zero_m1=zero_m1,
                        freeb_bsqvac_half=freeb_bsqvac_half_trial,
                        constraint_precond_diag=constraint_precond_diag,
                        constraint_tcon=constraint_tcon_override,
                        constraint_precond_active=constraint_precond_active,
                        constraint_tcon_active=constraint_tcon_active,
                        iter2=iter2,
                    )
                    _record_compute_force_timing("trial", t_trial_force_start, gcr2_t)
                    fsqr_t, fsqz_t, fsql_t = _fsq_from_norms(
                        norms_t,
                        gcr2_in=gcr2_t,
                        gcz2_in=gcz2_t,
                        gcl2_in=gcl2_t,
                    )
                    w_try = float(np.asarray(fsqr_t + fsqz_t + fsql_t))
                    w_try_ratio = w_try / max(w_curr, 1e-30) if np.isfinite(w_try) else float("inf")
                    if np.isfinite(w_try) and (w_try <= accept_ratio * max(w_curr, 1e-30)):
                        # Keep momentum consistent with the smaller step.
                        vRcc = alpha * vRcc
                        vRss = alpha * vRss
                        vZsc = alpha * vZsc
                        vZcs = alpha * vZcs
                        vLsc = alpha * vLsc
                        vLcs = alpha * vLcs
                        update_rms *= alpha
                        dt_eff *= alpha
                        break

            # Require (near) monotone improvement; otherwise fall back to the
            # restart/timestep control path.
            if np.isfinite(w_try) and (w_try <= accept_ratio * max(w_curr, 1e-30)):
                state = state_try
                step_status = "momentum"
                restart_reason = "none"
                huge_force_restart_count = 0
                restart_path = "momentum_accept"
            else:
                if use_direct_fallback:
                    # Try a small direct-force step (no momentum memory) before
                    # a full restart. This is an experimental parity path.
                    dt_direct = max(0.1 * dt_eff, 1e-12)
                    force_rms = float(
                        np.asarray(
                            jnp.sqrt(
                                jnp.mean(
                                    frcc_u * frcc_u
                                    + frss_u * frss_u
                                    + frsc_u * frsc_u
                                    + frcs_u * frcs_u
                                    + fzsc_u * fzsc_u
                                    + fzcs_u * fzcs_u
                                    + fzcc_u * fzcc_u
                                    + fzss_u * fzss_u
                                    + flsc_u * flsc_u
                                    + flcs_u * flcs_u
                                    + flcc_u * flcc_u
                                    + flss_u * flss_u
                                )
                            )
                        )
                    )
                    if np.isfinite(force_rms) and force_rms > 0.0:
                        dt_cap = max_update_rms / max(force_rms, 1e-30)
                        dt_direct = max(min(dt_direct, float(dt_cap)), 1e-12)
                    dR_dir = dt_direct * _mn_cos_to_signed(flip_sign * frcc_u, flip_sign * frss_u)
                    dZ_dir = dt_direct * _mn_sin_to_signed(flip_sign * fzsc_u, flip_sign * fzcs_u)
                    dL_dir = dt_direct * _mn_sin_to_signed(flip_sign * flsc_u, flip_sign * flcs_u)
                    if bool(cfg.lasym):
                        dR_sin_dir = dt_direct * _mn_sin_to_signed(flip_sign * frsc_u, flip_sign * frcs_u)
                        dZ_cos_dir = dt_direct * _mn_cos_to_signed(flip_sign * fzcc_u, flip_sign * fzss_u)
                        dL_cos_dir = dt_direct * _mn_cos_to_signed(flip_sign * flcc_u, flip_sign * flss_u)
                    else:
                        dR_sin_dir = jnp.zeros_like(dR_dir)
                        dZ_cos_dir = jnp.zeros_like(dR_dir)
                        dL_cos_dir = jnp.zeros_like(dR_dir)
                    state_dir = VMECState(
                        layout=state.layout,
                        Rcos=jnp.asarray(state.Rcos) + dR_dir,
                        Rsin=jnp.asarray(state.Rsin) + dR_sin_dir,
                        Zcos=jnp.asarray(state.Zcos) + dZ_cos_dir,
                        Zsin=jnp.asarray(state.Zsin) + dZ_dir,
                        Lcos=jnp.asarray(state.Lcos) + dL_cos_dir,
                        Lsin=jnp.asarray(state.Lsin) + dL_dir,
                    )
                    state_dir = _enforce_fixed_boundary_and_axis(
                        state_dir,
                        static,
                        edge_Rcos=edge_Rcos,
                        edge_Rsin=edge_Rsin,
                        edge_Zcos=edge_Zcos,
                        edge_Zsin=edge_Zsin,
                        enforce_edge=not bool(free_boundary_enabled),
                        enforce_lambda_axis=True,
                        idx00=idx00,
                    )
                    state_dir = _apply_vmec_lambda_axis_rules(state_dir)
                    freeb_bsqvac_half_dir = _freeb_bsqvac_half_for_trial_state(state_dir)
                    _, _, gcr2_d, gcz2_d, gcl2_d, _, _, norms_d = _compute_forces_iter(
                        state_dir,
                        include_edge=include_edge,
                        zero_m1=zero_m1,
                        freeb_bsqvac_half=freeb_bsqvac_half_dir,
                        constraint_precond_diag=constraint_precond_diag,
                        constraint_tcon=constraint_tcon_override,
                        constraint_precond_active=constraint_precond_active,
                        constraint_tcon_active=constraint_tcon_active,
                        iter2=iter2,
                    )
                    fsqr_d, fsqz_d, fsql_d = _fsq_from_norms(
                        norms_d,
                        gcr2_in=gcr2_d,
                        gcz2_in=gcz2_d,
                        gcl2_in=gcl2_d,
                    )
                    w_dir = float(np.asarray(fsqr_d + fsqz_d + fsql_d))
                    if np.isfinite(w_dir) and (w_dir <= 1.5 * max(w_curr, 1e-30)):
                        state = state_dir
                        (
                            vRcc,
                            vRss,
                            vZsc,
                            vZcs,
                            vLsc,
                            vLcs,
                            vRsc,
                            vRcs,
                            vZcc,
                            vZss,
                            vLcc,
                            vLss,
                        ) = _zero_velocity_blocks_like(
                            vRcc,
                            vRss,
                            vZsc,
                            vZcs,
                            vLsc,
                            vLcs,
                            vRsc,
                            vRcs,
                            vZcc,
                            vZss,
                            vLcc,
                            vLss,
                        )
                        step_status = "fallback_direct"
                        restart_reason = "none"
                        huge_force_restart_count = 0
                        restart_path = "fallback_direct"
                        update_rms = float(
                            np.asarray(
                                jnp.sqrt(
                                    jnp.mean(
                                        (dt_direct * frcc_u) ** 2
                                        + (dt_direct * frss_u) ** 2
                                        + (dt_direct * frsc_u) ** 2
                                        + (dt_direct * frcs_u) ** 2
                                        + (dt_direct * fzsc_u) ** 2
                                        + (dt_direct * fzcs_u) ** 2
                                        + (dt_direct * fzcc_u) ** 2
                                        + (dt_direct * fzss_u) ** 2
                                        + (dt_direct * flsc_u) ** 2
                                        + (dt_direct * flcs_u) ** 2
                                        + (dt_direct * flcc_u) ** 2
                                        + (dt_direct * flss_u) ** 2
                                    )
                                )
                            )
                        )
                        if adjoint_trace:
                            trace_entry["fallback_direct_dt"] = float(dt_direct)
                    else:
                        # Roll back state and zero velocity.
                        state = state_backup
                        vRcc, vRss, vZsc, vZcs, vLsc, vLcs = _zero_velocity_blocks_like(
                            vRcc, vRss, vZsc, vZcs, vLsc, vLcs
                        )
                        # Tighten displacement caps when restarting from
                        # catastrophic growth; otherwise dt_eff can remain
                        # stuck at the same limit.
                        max_coeff_delta_rms = max(0.5 * max_coeff_delta_rms, 1e-12)
                        max_update_rms = max(0.8 * max_update_rms, 1e-6)
                        if bool(probe_bad_jacobian) or (not np.isfinite(w_try)):
                            time_step = max(restart_badjac_factor * time_step, 1e-12)
                            ijacob += 1
                            restart_reason = "bad_jacobian"
                            step_status = "restart_bad_jacobian"
                            restart_path = "catastrophic_nonfinite"
                        else:
                            time_step = max(time_step / restart_badprog_factor, 1e-12)
                            restart_reason = "bad_progress"
                            step_status = "restart_bad_progress"
                            restart_path = "catastrophic_growth"
                        # Adjust time_step at reset milestones.
                        if ijacob in (25, 50):
                            scale = 0.98 if ijacob < 50 else 0.96
                            time_step = max(scale * float(step_size), 1e-12)
                        bad_resets += 1
                        iter1 = iter2
                        freeb_controls_cached = None
                        fsq_prev = fsq_prev_before
                        fsq0_prev = fsq0_prev_before
                        inv_tau = [0.15 / time_step] * k_ndamp
                        update_rms = 0.0
                        if bool(vmec2000_control):
                            vmec2000_cache_valid = False
                            cache_precond_diag = None
                            cache_tcon = None
                            cache_norms = None
                            cache_rz_scale = None
                            cache_l_scale = None
                            cache_rz_norm = None
                            cache_f_norm1 = None
                            cache_prec_rz_mats = None
                            cache_prec_rz_jmax = None
                            cache_prec_lam_prec = None
                            cache_prec_faclam = None
                            cache_prec_lam_debug = None
                else:
                    # Roll back state and zero velocity.
                    state = state_backup
                    vRcc, vRss, vZsc, vZcs, vLsc, vLcs = _zero_velocity_blocks_like(vRcc, vRss, vZsc, vZcs, vLsc, vLcs)
                    # Tighten displacement caps when restarting from catastrophic
                    # growth; otherwise dt_eff can remain stuck at the same limit.
                    max_coeff_delta_rms = max(0.5 * max_coeff_delta_rms, 1e-12)
                    max_update_rms = max(0.8 * max_update_rms, 1e-6)
                    if bool(probe_bad_jacobian) or (not np.isfinite(w_try)):
                        time_step = max(restart_badjac_factor * time_step, 1e-12)
                        ijacob += 1
                        restart_reason = "bad_jacobian"
                        step_status = "restart_bad_jacobian"
                        restart_path = "catastrophic_nonfinite"
                    else:
                        time_step = max(time_step / restart_badprog_factor, 1e-12)
                        restart_reason = "bad_progress"
                        step_status = "restart_bad_progress"
                        restart_path = "catastrophic_growth"
                    # Adjust time_step at reset milestones.
                    if ijacob in (25, 50):
                        scale = 0.98 if ijacob < 50 else 0.96
                        time_step = max(scale * float(step_size), 1e-12)
                    bad_resets += 1
                    iter1 = iter2
                    freeb_controls_cached = None
                    fsq_prev = fsq_prev_before
                    fsq0_prev = fsq0_prev_before
                    inv_tau = [0.15 / time_step] * k_ndamp
                    update_rms = 0.0
                    if not bool(vmec2000_control):
                        vmec2000_cache_valid = False
                        cache_precond_diag = None
                        cache_tcon = None
                        cache_norms = None
                        cache_rz_scale = None
                        cache_l_scale = None
                        cache_rz_norm = None
                        cache_f_norm1 = None
                        cache_prec_rz_mats = None
                        cache_prec_rz_jmax = None
                        cache_prec_lam_prec = None
                        cache_prec_faclam = None
                        cache_prec_lam_debug = None
            if timing_enabled and t_state_update_start is not None:
                try:
                    if has_jax():
                        jax.block_until_ready(state.Rcos)
                except Exception:
                    pass
                timing_stats["update_state"] += time.perf_counter() - float(t_state_update_start)
            t_trace_finalize_start = time.perf_counter() if timing_enabled and adjoint_trace else None
            if adjoint_trace:
                trace_entry.update(
                    {
                        "step_status": str(step_status),
                        "restart_reason": str(restart_reason),
                        "restart_path": str(restart_path),
                        "time_step": float(time_step),
                        "flip_sign": float(flip_sign),
                        "limit_update_rms": bool(limit_update_rms),
                    }
                )
                if adjoint_trace_mode == "full":
                    trace_entry.update(
                        {
                            "dt_eff": float(dt_eff),
                            "w_curr": float(w_curr),
                            "w_try": float(w_try),
                            "w_try_ratio": float(w_try_ratio),
                            "b1": float(b1),
                            "fac": float(fac),
                            "force_scale": float(force_scale),
                            "update_rms_preclip": None if update_rms_preclip is None else float(update_rms_preclip),
                            "update_rms_postclip": None if update_rms is None else float(update_rms),
                            "update_rms_scale": float(scl),
                            "state_post": state,
                            "vRcc_after": np.asarray(vRcc),
                            "vRss_after": np.asarray(vRss),
                            "vZsc_after": np.asarray(vZsc),
                            "vZcs_after": np.asarray(vZcs),
                            "vLsc_after": np.asarray(vLsc),
                            "vLcs_after": np.asarray(vLcs),
                            "vRsc_after": np.asarray(vRsc),
                            "vRcs_after": np.asarray(vRcs),
                            "vZcc_after": np.asarray(vZcc),
                            "vZss_after": np.asarray(vZss),
                            "vLcc_after": np.asarray(vLcc),
                            "vLss_after": np.asarray(vLss),
                        }
                    )
                adjoint_step_trace_history.append(trace_entry)
            if timing_enabled and t_trace_finalize_start is not None:
                timing_stats["update_trace_finalize"] += time.perf_counter() - float(t_trace_finalize_start)
            if timing_enabled and t_update_start is not None:
                try:
                    if has_jax():
                        jax.block_until_ready(state.Rcos)
                except Exception:
                    pass
                timing_stats["update"] += time.perf_counter() - float(t_update_start)
            timing_stats["iterations"] += 1
            if track_history:
                step_history.append(float(dt_eff))
                w_curr_history.append(float(w_curr))
                w_try_history.append(float(w_try))
                w_try_ratio_history.append(float(w_try_ratio))
                restart_path_history.append(str(restart_path))
        else:
            accepted = False
            step_status = "rejected"
            step_factor = 1.0
            vRcc_best, vRss_best = vRcc, vRss
            vZsc_best, vZcs_best = vZsc, vZcs
            vLsc_best, vLcs_best = vLsc, vLcs
            vRsc_best, vRcs_best = vRsc, vRcs
            vZcc_best, vZss_best = vZcc, vZss
            vLcc_best, vLss_best = vLcc, vLss
            state_best = state
            dt_eff = float(time_step)
            update_rms = 0.0
            w_curr = fsqr_f + fsqz_f + fsql_f

            for _bt in range(6):
                dt_try = time_step * step_factor
                vRcc_try = fac * (b1 * vRcc + dt_try * (flip_sign * jnp.asarray(frcc_u)))
                vRss_try = fac * (b1 * vRss + dt_try * (flip_sign * jnp.asarray(frss_u)))
                vRsc_try = fac * (b1 * vRsc + dt_try * (flip_sign * jnp.asarray(frsc_u)))
                vRcs_try = fac * (b1 * vRcs + dt_try * (flip_sign * jnp.asarray(frcs_u)))
                vZsc_try = fac * (b1 * vZsc + dt_try * (flip_sign * jnp.asarray(fzsc_u)))
                vZcs_try = fac * (b1 * vZcs + dt_try * (flip_sign * jnp.asarray(fzcs_u)))
                vZcc_try = fac * (b1 * vZcc + dt_try * (flip_sign * jnp.asarray(fzcc_u)))
                vZss_try = fac * (b1 * vZss + dt_try * (flip_sign * jnp.asarray(fzss_u)))
                vLsc_try = fac * (b1 * vLsc + dt_try * (flip_sign * jnp.asarray(flsc_u)))
                vLcs_try = fac * (b1 * vLcs + dt_try * (flip_sign * jnp.asarray(flcs_u)))
                vLcc_try = fac * (b1 * vLcc + dt_try * (flip_sign * jnp.asarray(flcc_u)))
                vLss_try = fac * (b1 * vLss + dt_try * (flip_sign * jnp.asarray(flss_u)))

                dR_try = dt_try * _mn_cos_to_signed(vRcc_try, vRss_try)
                dZ_try = dt_try * _mn_sin_to_signed(vZsc_try, vZcs_try)
                dL_try = dt_try * _mn_sin_to_signed(vLsc_try, vLcs_try)
                if bool(cfg.lasym):
                    dR_sin_try = dt_try * _mn_sin_to_signed(vRsc_try, vRcs_try)
                    dZ_cos_try = dt_try * _mn_cos_to_signed(vZcc_try, vZss_try)
                    dL_cos_try = dt_try * _mn_cos_to_signed(vLcc_try, vLss_try)
                else:
                    dR_sin_try = jnp.zeros_like(dR_try)
                    dZ_cos_try = jnp.zeros_like(dR_try)
                    dL_cos_try = jnp.zeros_like(dR_try)

                state_try = VMECState(
                    layout=state.layout,
                    Rcos=jnp.asarray(state.Rcos) + dR_try,
                    Rsin=jnp.asarray(state.Rsin) + dR_sin_try,
                    Zcos=jnp.asarray(state.Zcos) + dZ_cos_try,
                    Zsin=jnp.asarray(state.Zsin) + dZ_try,
                    Lcos=jnp.asarray(state.Lcos) + dL_cos_try,
                    Lsin=jnp.asarray(state.Lsin) + dL_try,
                )
                state_try = _enforce_fixed_boundary_and_axis(
                    state_try,
                    static,
                    edge_Rcos=edge_Rcos,
                    edge_Rsin=edge_Rsin,
                    edge_Zcos=edge_Zcos,
                    edge_Zsin=edge_Zsin,
                    enforce_edge=not bool(free_boundary_enabled),
                    enforce_lambda_axis=True,
                    idx00=idx00,
                )
                state_try = _apply_vmec_lambda_axis_rules(state_try)
                freeb_bsqvac_half_trial = _freeb_bsqvac_half_for_trial_state(state_try)
                t_backtracking_force_start = time.perf_counter() if timing_detail_enabled else None
                _, _, gcr2_t, gcz2_t, gcl2_t, _, _, norms_t = _compute_forces_iter(
                    state_try,
                    include_edge=include_edge,
                    zero_m1=zero_m1,
                    freeb_bsqvac_half=freeb_bsqvac_half_trial,
                    constraint_precond_diag=constraint_precond_diag,
                    constraint_tcon=constraint_tcon_override,
                    constraint_precond_active=constraint_precond_active,
                    constraint_tcon_active=constraint_tcon_active,
                    iter2=iter2,
                )
                _record_compute_force_timing("backtracking", t_backtracking_force_start, gcr2_t)
                fsqr_t, fsqz_t, fsql_t = _fsq_from_norms(
                    norms_t,
                    gcr2_in=gcr2_t,
                    gcz2_in=gcz2_t,
                    gcl2_in=gcl2_t,
                )
                w_try = float(np.asarray(fsqr_t + fsqz_t + fsql_t))
                if np.isfinite(w_try) and (w_try <= 1.05 * w_curr):
                    accepted = True
                    step_status = "momentum"
                    state_best = state_try
                    vRcc_best, vRss_best = vRcc_try, vRss_try
                    vZsc_best, vZcs_best = vZsc_try, vZcs_try
                    vLsc_best, vLcs_best = vLsc_try, vLcs_try
                    vRsc_best, vRcs_best = vRsc_try, vRcs_try
                    vZcc_best, vZss_best = vZcc_try, vZss_try
                    vLcc_best, vLss_best = vLcc_try, vLss_try
                    dt_eff = float(dt_try)
                    update_rms = float(
                        np.asarray(
                            jnp.sqrt(
                                jnp.mean(
                                    (dt_try * vRcc_try) ** 2
                                    + (dt_try * vRss_try) ** 2
                                    + (dt_try * vRsc_try) ** 2
                                    + (dt_try * vRcs_try) ** 2
                                    + (dt_try * vZsc_try) ** 2
                                    + (dt_try * vZcs_try) ** 2
                                    + (dt_try * vZcc_try) ** 2
                                    + (dt_try * vZss_try) ** 2
                                    + (dt_try * vLsc_try) ** 2
                                    + (dt_try * vLcs_try) ** 2
                                    + (dt_try * vLcc_try) ** 2
                                    + (dt_try * vLss_try) ** 2
                                )
                            )
                        )
                    )
                    break
                step_factor *= 0.5

            state = state_best
            vRcc, vRss = vRcc_best, vRss_best
            vZsc, vZcs = vZsc_best, vZcs_best
            vLsc, vLcs = vLsc_best, vLcs_best
            vRsc, vRcs = vRsc_best, vRcs_best
            vZcc, vZss = vZcc_best, vZss_best
            vLcc, vLss = vLcc_best, vLss_best
            if not accepted:
                # No acceptable update was found; damp velocity to avoid runaway.
                (
                    vRcc,
                    vRss,
                    vRsc,
                    vRcs,
                    vZsc,
                    vZcs,
                    vZcc,
                    vZss,
                    vLsc,
                    vLcs,
                    vLcc,
                    vLss,
                ) = _scale_velocity_blocks(0.5, vRcc, vRss, vRsc, vRcs, vZsc, vZcs, vZcc, vZss, vLsc, vLcs, vLcc, vLss)
                dt_eff = float(step_size * step_factor)
                update_rms = 0.0
                step_status = "rejected"
            timing_stats["iterations"] += 1
            if track_history:
                step_history.append(dt_eff)
                restart_reason = "none"
                w_curr_history.append(float(w_curr))
                w_try_history.append(float("nan"))
                w_try_ratio_history.append(float("nan"))
                restart_path_history.append("non_strict")
        t_iteration_post_update_start = time.perf_counter() if timing_enabled else None
        _dump_evolve_trace(
            iter2=int(iter2),
            iter1=int(iter1),
            stage="post",
            fsq1_val=float(fsq1),
            fsq_prev_val=float(fsq_prev_before),
            time_step_val=float(time_step),
            dtau_val=float(dtau),
            b1_val=float(b1),
            fac_val=float(fac),
            state_val=state,
            vRcc_val=vRcc,
            vRss_val=vRss,
            vZsc_val=vZsc,
            vZcs_val=vZcs,
            vLsc_val=vLsc,
            vLcs_val=vLcs,
            vRsc_val=vRsc,
            vRcs_val=vRcs,
            vZcc_val=vZcc,
            vZss_val=vZss,
            vLcc_val=vLcc,
            vLss_val=vLss,
            frcc_val=frcc_u,
            frss_val=frss_u,
            fzsc_val=fzsc_u,
            fzcs_val=fzcs_u,
            flsc_val=flsc_u,
            flcs_val=flcs_u,
            frsc_val=frsc_u,
            frcs_val=frcs_u,
            fzcc_val=fzcc_u,
            fzss_val=fzss_u,
            flcc_val=flcc_u,
            flss_val=flss_u,
        )
        _maybe_dump_xc(
            state=state,
            vRcc=vRcc,
            vRss=vRss,
            vZsc=vZsc,
            vZcs=vZcs,
            vLsc=vLsc,
            vLcs=vLcs,
            vRsc=vRsc,
            vRcs=vRcs,
            vZcc=vZcc,
            vZss=vZss,
            vLcc=vLcc,
            vLss=vLss,
            static=static,
            iter_idx=int(iter2),
        )
        if track_history:
            dt_eff_history.append(float(dt_eff))
            update_rms_history.append(update_rms_j if bool(strict_update) else float(update_rms))
        if verbose:
            if bool(vmec2000_control) and bool(verbose_vmec2000_table):
                if _should_print_vmec2000(int(iter2), int(max_iter)):
                    fsqr1_f, fsqz1_f, fsql1_f = _precond_diag_floats()
                    _print_vmec2000_iter_row(
                        iter_idx=int(iter2),
                        fsqr=fsqr_f,
                        fsqz=fsqz_f,
                        fsql=fsql_f,
                        fsqr1=fsqr1_f,
                        fsqz1=fsqz1_f,
                        fsql1=fsql1_f,
                        delt0r=float(time_step),
                        r00=float(r00_last),
                        w_mhd=float(w_vmec_last),
                        z00=float(z00_last),
                    )
            else:
                fsqr1_f, fsqz1_f, fsql1_f = _precond_diag_floats()
                update_rms_print = _update_rms_float() if bool(strict_update) else float(update_rms)
                print(
                    f"[solve_fixed_boundary_residual_iter] iter={it:03d} "
                    f"dt_eff={dt_eff:.3e} update_rms={update_rms_print:.3e} "
                    f"fsqr1={fsqr1_f:.3e} fsqz1={fsqz1_f:.3e} fsql1={fsql1_f:.3e} "
                    f"step_status={step_status}",
                    flush=True,
                )
        if track_history:
            _append_residual_iter_terminal_history(
                step_status=step_status,
                restart_reason=restart_reason,
                pre_restart_reason=pre_restart_reason,
                time_step=float(time_step),
                res0=float(res0),
                res1=float(res1),
                fsq_prev=float(fsq_prev),
                bad_growth_streak=int(bad_growth_streak),
                iter1=int(iter1),
                iter2=int(iter2),
                fsqr=fsqr_f,
                fsqz=fsqz_f,
                fsql=fsql_f,
                step_status_history=step_status_history,
                restart_reason_history=restart_reason_history,
                pre_restart_reason_history=pre_restart_reason_history,
                time_step_history=time_step_history,
                res0_history=res0_history,
                res1_history=res1_history,
                fsq_prev_history=fsq_prev_history,
                bad_growth_streak_history=bad_growth_streak_history,
                iter1_history=iter1_history,
                iter2_history=iter2_history,
                grad_rms_history=grad_rms_history,
                free_boundary_enabled=free_boundary_enabled,
                freeb_ivac=freeb_ivac,
                freeb_ivacskip=freeb_ivacskip,
                freeb_reused=freeb_reused,
                freeb_solve_time=freeb_solve_time,
                freeb_sample_time=freeb_sample_time,
                freeb_ivac_history=freeb_ivac_history,
                freeb_ivacskip_history=freeb_ivacskip_history,
                freeb_full_update_history=freeb_full_update_history,
                freeb_nestor_reused_history=freeb_nestor_reused_history,
                freeb_nestor_solve_time_history=freeb_nestor_solve_time_history,
                freeb_nestor_sample_time_history=freeb_nestor_sample_time_history,
            )
        # VMEC eqsolve behavior: when `ivac==1`, print turn-on and promote to
        # `ivac=2` for subsequent iterations.
        if free_boundary_enabled and int(freeb_ivac) == 1:
            if verbose and bool(verbose_vmec2000_table):
                print(f"\n  VACUUM PRESSURE TURNED ON AT {int(iter2):4d} ITERATIONS\n", flush=True)
            freeb_ivac = int(freeb_ivac) + 1
        skip_time_control = False
        if timing_enabled and t_iteration_post_update_start is not None:
            timing_stats["iteration_post_update"] += time.perf_counter() - float(t_iteration_post_update_start)

    t_finalize_start = time.perf_counter() if timing_enabled else None
    final_fsqr_report = float(fsqr_f)
    final_fsqz_report = float(fsqz_f)
    final_fsql_report = float(fsql_f)
    final_residual_recomputed = False
    final_pre_update_fsqr = float(fsqr_f)
    final_pre_update_fsqz = float(fsqz_f)
    final_pre_update_fsql = float(fsql_f)
    final_nestor_model = str(freeb_last_model)
    final_nestor_diagnostics = dict(freeb_last_diagnostics)
    final_vacuum_stub = not bool(str(final_nestor_model).strip() and str(final_nestor_model) != "none")
    final_bsqvac_half_current = freeb_bsqvac_half_current
    final_nestor_recompute_attempted = False
    final_nestor_recompute_failed = False
    final_nestor_sample_time_s = 0.0
    final_nestor_solve_time_s = 0.0
    if bool(free_boundary_enabled and freeb_couple_edge) and not final_vacuum_stub:
        final_nestor_recompute_attempted = True
        try:
            nestor_final, _freeb_nestor_runtime_final = nestor_external_only_step(
                state=state,
                static=static,
                ivac=1,
                ivacskip=0,
                iter_idx=None,
                runtime=freeb_nestor_runtime,
                extcur=tuple(getattr(static, "free_boundary_extcur", ()) or ()),
                plascur=float(freeb_plascur),
                external_field_provider_kind=external_field_provider_kind,
                external_field_provider_static=external_field_provider_static,
                external_field_provider_params=external_field_provider_params,
            )
            final_nestor_sample_time_s = float(getattr(nestor_final, "sample_time_s", 0.0))
            final_nestor_solve_time_s = float(getattr(nestor_final, "solve_time_s", 0.0))
            final_nestor_model = str(getattr(nestor_final, "model", final_nestor_model))
            diag_final = getattr(nestor_final, "diagnostics", None)
            if isinstance(diag_final, dict):
                final_nestor_diagnostics = dict(diag_final)
            bsqvac_edge_final = np.asarray(nestor_final.vac_total.bsqvac, dtype=float)
            if (
                bsqvac_edge_final.ndim == 2
                and int(bsqvac_edge_final.shape[1]) == 1
                and int(getattr(static.cfg, "nzeta", 1)) > 1
            ):
                bsqvac_edge_final = np.repeat(bsqvac_edge_final, int(static.cfg.nzeta), axis=1)
            final_bsqvac_half_current = bsqvac_edge_final
            final_vacuum_stub = False
        except Exception:
            final_nestor_recompute_failed = True
            final_bsqvac_half_current = freeb_bsqvac_half_current
    if bool(free_boundary_enabled) and final_bsqvac_half_current is not None:
        try:
            _, _, gcr2_final, gcz2_final, gcl2_final, _, _, norms_final = _compute_forces_iter(
                state,
                include_edge=bool(include_edge),
                include_edge_residual=True,
                zero_m1=zero_m1,
                freeb_bsqvac_half=final_bsqvac_half_current,
                constraint_precond_diag=constraint_precond_diag,
                constraint_tcon=constraint_tcon_override,
                constraint_precond_active=constraint_precond_active,
                constraint_tcon_active=constraint_tcon_active,
                iter2=last_iter2,
            )
            fsqr_final, fsqz_final, fsql_final = _fsq_from_norms(
                norms_final,
                gcr2_in=gcr2_final,
                gcz2_in=gcz2_final,
                gcl2_in=gcl2_final,
            )
            final_fsqr_report, final_fsqz_report, final_fsql_report = _device_get_floats(
                fsqr_final,
                fsqz_final,
                fsql_final,
            )
            final_residual_recomputed = True
        except Exception:
            final_fsqr_report = float(fsqr_f)
            final_fsqz_report = float(fsqz_f)
            final_fsql_report = float(fsql_f)
    converged_strict_final, converged_total_final, _ = _residual_convergence_flags(
        fsqr=final_fsqr_report,
        fsqz=final_fsqz_report,
        fsql=final_fsql_report,
        ftol=ftol,
        fsq_total_target=fsq_total_target,
    )
    diag: Dict[str, Any] = {
        "ftol": ftol,
        "requested_ftol": float(ftol),
        "gamma": gamma,
        "step_size": float(step_size),
        "precond_radial_alpha": float(precond_radial_alpha),
        "precond_lambda_alpha": float(precond_lambda_alpha),
        "strict_update": bool(strict_update),
        "reference_mode": bool(reference_mode),
        "use_restart_triggers": bool(use_restart_triggers),
        "use_direct_fallback": bool(use_direct_fallback),
        "max_update_rms": float(max_update_rms),
        "converged": bool(converged),
        "converged_strict": bool(converged_strict_final),
        "converged_by_total_fsq": bool(converged_total_final),
        "final_fsqr": float(final_fsqr_report),
        "final_fsqz": float(final_fsqz_report),
        "final_fsql": float(final_fsql_report),
        "pre_update_final_fsqr": float(final_pre_update_fsqr),
        "pre_update_final_fsqz": float(final_pre_update_fsqz),
        "pre_update_final_fsql": float(final_pre_update_fsql),
        "final_residual_recomputed_on_accepted_state": bool(final_residual_recomputed),
        "badjac_use_state": bool(badjac_use_state),
        "badjac_mode": badjac_mode,
        "badjac_state_probe": bool(badjac_state_probe),
        "badjac_initial_state_probe_iters": int(badjac_initial_state_probe_iters),
        "light_history": bool(light_history),
        "resume_state_mode": str(resume_state_mode),
        "fsq_total_target": fsq_total_target,
        "ijacob": int(ijacob),
        "bad_resets": int(bad_resets),
        "iter1_final": int(iter1),
        "res0": float(res0),
        "step_status_history": np.asarray(step_status_history, dtype=object),
        "restart_reason_history": np.asarray(restart_reason_history, dtype=object),
        "pre_restart_reason_history": np.asarray(pre_restart_reason_history, dtype=object),
        "time_step_history": np.asarray(time_step_history, dtype=float),
        "res0_history": np.asarray(res0_history, dtype=float),
        "res1_history": np.asarray(res1_history, dtype=float),
        "fsq_prev_history": np.asarray(fsq_prev_history, dtype=float),
        "bad_growth_streak_history": np.asarray(bad_growth_streak_history, dtype=int),
        "iter1_history": np.asarray(iter1_history, dtype=int),
        "iter2_history": np.asarray(iter2_history, dtype=int),
        "bcovar_update_history": np.asarray(bcovar_update_history, dtype=int),
        "include_edge_history": np.asarray(include_edge_history, dtype=int),
        "zero_m1_history": np.asarray(zero_m1_history, dtype=int),
        "dt_eff_history": np.asarray(dt_eff_history, dtype=float),
        "update_rms_history": _scalar_history_array(update_rms_history),
        "w_curr_history": np.asarray(w_curr_history, dtype=float),
        "w_try_history": np.asarray(w_try_history, dtype=float),
        "w_try_ratio_history": np.asarray(w_try_ratio_history, dtype=float),
        "restart_path_history": np.asarray(restart_path_history, dtype=object),
        "adjoint_step_trace": adjoint_step_trace_history,
        "min_tau_history": np.asarray(min_tau_history, dtype=float),
        "max_tau_history": np.asarray(max_tau_history, dtype=float),
        "bad_jacobian_history": np.asarray(bad_jacobian_history, dtype=int),
        "r00_history": np.asarray(r00_history, dtype=float),
        "z00_history": np.asarray(z00_history, dtype=float),
        "wb_history": np.asarray(wb_history, dtype=float),
        "wp_history": np.asarray(wp_history, dtype=float),
        "w_vmec_history": np.asarray(w_vmec_history, dtype=float),
        "fsq1_history": _scalar_history_array(fsq1_history),
        "fsqr1_history": _scalar_history_array(fsqr1_history),
        "fsqz1_history": _scalar_history_array(fsqz1_history),
        "fsql1_history": _scalar_history_array(fsql1_history),
        "rz_norm_history": _scalar_history_array(rz_norm_history),
        "f_norm1_history": _scalar_history_array(f_norm1_history),
        "gcr2_p_history": _scalar_history_array(gcr2_p_history),
        "gcz2_p_history": _scalar_history_array(gcz2_p_history),
        "gcl2_p_history": _scalar_history_array(gcl2_p_history),
        "free_boundary": {
            "enabled": bool(free_boundary_enabled),
            "nvacskip": int(freeb_nvacskip),
            "nvskip0": int(freeb_nvskip0),
            "ivac": int(freeb_ivac),
            "ivacskip": int(freeb_ivacskip),
            "couple_edge": bool(freeb_couple_edge),
            "nestor_model": str(final_nestor_model),
            "vacuum_stub": bool(final_vacuum_stub),
            "activate_fsq": None if free_boundary_activate_fsq is None else float(free_boundary_activate_fsq),
            "plascur": float(freeb_plascur),
            "last_nestor_diagnostics": dict(final_nestor_diagnostics),
            "final_nestor_recompute_attempted": bool(final_nestor_recompute_attempted),
            "final_nestor_recompute_failed": bool(final_nestor_recompute_failed),
            "final_nestor_sample_time_s": float(final_nestor_sample_time_s),
            "final_nestor_solve_time_s": float(final_nestor_solve_time_s),
        },
        "freeb_ivac_history": np.asarray(freeb_ivac_history, dtype=int),
        "freeb_ivacskip_history": np.asarray(freeb_ivacskip_history, dtype=int),
        "freeb_full_update_history": np.asarray(freeb_full_update_history, dtype=int),
        "freeb_nestor_reused_history": np.asarray(freeb_nestor_reused_history, dtype=int),
        "freeb_nestor_source_reused_history": np.asarray(freeb_nestor_source_reused_history, dtype=int),
        "freeb_nestor_provider_allows_source_reuse_history": np.asarray(
            freeb_nestor_provider_allows_source_reuse_history, dtype=int
        ),
        "freeb_nestor_bnormal_rms_history": np.asarray(freeb_nestor_bnormal_rms_history, dtype=float),
        "freeb_nestor_gsource_rms_history": np.asarray(freeb_nestor_gsource_rms_history, dtype=float),
        "freeb_nestor_bsqvac_rms_history": np.asarray(freeb_nestor_bsqvac_rms_history, dtype=float),
        "freeb_nestor_solve_time_history": np.asarray(freeb_nestor_solve_time_history, dtype=float),
        "freeb_nestor_sample_time_history": np.asarray(freeb_nestor_sample_time_history, dtype=float),
        "freeb_nestor_trial_reused_history": np.asarray(freeb_nestor_trial_reused_history, dtype=int),
        "freeb_nestor_trial_solve_time_history": np.asarray(freeb_nestor_trial_solve_time_history, dtype=float),
        "freeb_nestor_trial_sample_time_history": np.asarray(freeb_nestor_trial_sample_time_history, dtype=float),
        "freeb_nestor_trial_failed_history": np.asarray(freeb_nestor_trial_failed_history, dtype=int),
    }
    if timing_enabled:
        if t_iteration_loop_start is not None and t_finalize_start is not None:
            timing_stats["iteration_loop"] = float(t_finalize_start) - float(t_iteration_loop_start)
        if t_finalize_start is not None:
            timing_stats["finalize"] = time.perf_counter() - float(t_finalize_start)
        timing_report = _build_residual_iter_timing_report(
            timing_stats,
            solve_total_s=float(time.perf_counter() - float(_solve_wall_start)),
            timing_detail_enabled=bool(timing_detail_enabled),
        )
        timing_stats["iteration_loop_unattributed"] = float(timing_report["iteration_loop_unattributed_s"])
        diag["timing"] = timing_report
        try:
            print(
                _format_residual_iter_timing_message(
                    timing_report,
                    timing_detail_enabled=bool(timing_detail_enabled),
                ),
                flush=True,
            )
        except Exception:
            pass
    resume_state_payload = None
    if resume_state_mode != "none":
        resume_state_base = _build_resume_state_base(
            time_step=time_step,
            inv_tau=inv_tau,
            fsq_prev=fsq_prev,
            fsq0_prev=fsq0_prev,
            flip_sign=flip_sign,
            iter1=iter1,
            last_iter2=last_iter2,
            ijacob=ijacob,
            bad_resets=bad_resets,
            res0=res0,
            res1=res1,
            prev_rz_fsq=prev_rz_fsq,
            bad_growth_streak=bad_growth_streak,
            huge_force_restart_count=huge_force_restart_count,
            vmec2000_cache_valid=vmec2000_cache_valid,
            freeb_ivac=freeb_ivac,
            freeb_ivacskip=freeb_ivacskip,
            freeb_nvacskip=freeb_nvacskip,
            freeb_nvskip0=freeb_nvskip0,
            freeb_last_model=freeb_last_model,
            freeb_nestor_runtime=freeb_nestor_runtime,
        )
        resume_state_heavy = None
        if resume_state_mode == "full":
            resume_state_heavy = {
                "vRcc": np.asarray(vRcc),
                "vRss": np.asarray(vRss),
                "vZsc": np.asarray(vZsc),
                "vZcs": np.asarray(vZcs),
                "vLsc": np.asarray(vLsc),
                "vLcs": np.asarray(vLcs),
                "vRsc": np.asarray(vRsc),
                "vRcs": np.asarray(vRcs),
                "vZcc": np.asarray(vZcc),
                "vZss": np.asarray(vZss),
                "vLcc": np.asarray(vLcc),
                "vLss": np.asarray(vLss),
                "state_checkpoint": state_checkpoint,
                "cache_precond_diag": cache_precond_diag,
                "cache_tcon": cache_tcon,
                "cache_norms": cache_norms,
                "cache_rz_scale": cache_rz_scale,
                "cache_l_scale": cache_l_scale,
                "cache_rz_norm": cache_rz_norm,
                "cache_f_norm1": cache_f_norm1,
                "cache_prec_rz_mats": cache_prec_rz_mats,
                "cache_prec_rz_jmax": cache_prec_rz_jmax,
                "cache_prec_lam_prec": cache_prec_lam_prec,
                "cache_prec_faclam": cache_prec_faclam,
                "cache_prec_lam_debug": cache_prec_lam_debug,
                "cache_constraint_rcon0": cache_constraint_rcon0,
                "cache_constraint_zcon0": cache_constraint_zcon0,
            }
        resume_state_payload = _pack_resume_state(resume_state_base, resume_state_heavy)
    diag["resume_state"] = resume_state_payload
    result = _attach_freeb_diag(
        SolveVmecResidualResult(
            state=state,
            n_iter=len(w_history) - 1,
            w_history=np.asarray(w_history, dtype=float),
            fsqr2_history=np.asarray(fsqr2_history, dtype=float),
            fsqz2_history=np.asarray(fsqz2_history, dtype=float),
            fsql2_history=np.asarray(fsql2_history, dtype=float),
            grad_rms_history=np.asarray(grad_rms_history, dtype=float),
            step_history=np.asarray(step_history, dtype=float),
            diagnostics=diag,
        )
    )
    if bool(return_final_force_payload) and bool(converged):
        try:
            object.__setattr__(result, "_final_force_payload", k)
        except Exception:
            pass
    return result


def first_step_diagnostics(
    state0: VMECState,
    static,
    *,
    indata,
    signgs: int,
    step_size: float | None = None,
    include_constraint_force: bool = True,
    apply_m1_constraints: bool = True,
    precond_radial_alpha: float = 0.5,
    precond_lambda_alpha: float = 0.5,
    mode_diag_exponent: float = 1.0,
    include_edge: bool = True,
    zero_m1: bool = True,
    use_axisymmetric_preconditioner: bool = False,
) -> Dict[str, Any]:
    """Return a first-step diagnostic bundle (single force/precondition/update eval).

    This computes the initial forces, preconditioned residuals, time-step
    scalings, and the resulting first-step coefficient updates without
    running an iterative solve.
    """
    if not has_jax():
        raise ImportError("first_step_diagnostics requires JAX (jax + jaxlib)")

    from .energy import flux_profiles_from_indata
    from .static import build_static
    from .vmec_forces import vmec_forces_rz_from_wout, vmec_residual_internal_from_kernels
    from .vmec_residue import (
        vmec_apply_m1_constraints,
        vmec_apply_scalxc_to_tomnsps,
        vmec_force_norms_from_bcovar_dynamic,
        vmec_gcx2_from_tomnsps,
        vmec_rz_norm_from_state,
        vmec_scalxc_from_s,
        vmec_wint_from_trig,
        vmec_zero_m1_zforce,
    )
    from .vmec_tomnsp import TomnspsRZL, vmec_angle_grid, vmec_trig_tables

    signgs = int(signgs)
    cfg = static.cfg
    grid_vmec = vmec_angle_grid(
        ntheta=int(cfg.ntheta),
        nzeta=int(cfg.nzeta),
        nfp=int(cfg.nfp),
        lasym=bool(cfg.lasym),
    )
    static_vmec = build_static(cfg, grid=grid_vmec)
    s = jnp.asarray(static_vmec.s)
    flux = flux_profiles_from_indata(indata, s, signgs=signgs)
    chipf_wout = jnp.asarray(flux.chipf)
    phips = jnp.asarray(flux.phips)
    if phips.shape[0] >= 1:
        phips = phips.at[0].set(0.0)
    from .boundary import boundary_from_indata

    boundary = boundary_from_indata(indata, static_vmec.modes)
    idx00 = _mode00_index(static_vmec.modes)
    r00 = float(np.asarray(boundary.R_cos)[int(idx00)]) if int(idx00) >= 0 else float(np.asarray(boundary.R_cos)[0])
    gamma = float(indata.get_float("GAMMA", 0.0))
    lrfp = bool(indata.get_bool("LRFP", False))
    chips = _half_mesh_from_full_mesh(chipf_wout) if lrfp else None
    mass = _mass_half_mesh_from_indata(
        indata=indata,
        s_full=s,
        phips=phips,
        r00=r00,
        gamma=gamma,
        lrfp=lrfp,
        chips=chips,
    )
    pres = _pressure_half_mesh_from_indata(indata=indata, s_full=s)
    ncurr = int(indata.get_int("NCURR", 0))
    icurv = _icurv_full_mesh_from_indata(indata=indata, s_full=s, signgs=signgs)
    phipf_internal, chipf_internal, chips_eff = _vmec_force_flux_profiles(
        phipf=jnp.asarray(flux.phipf),
        chipf=chipf_wout,
        signgs=signgs,
        flux_is_internal=True,
    )

    wout_like = _WoutLikeVmecForces(
        nfp=int(cfg.nfp),
        mpol=int(cfg.mpol),
        ntor=int(cfg.ntor),
        lasym=bool(cfg.lasym),
        signgs=signgs,
        phipf=jnp.asarray(flux.phipf),
        phips=phips,
        chipf=chipf_wout,
        pres=pres,
        mass=mass,
        gamma=gamma,
        ncurr=ncurr,
        lcurrent=True,
        icurv=icurv,
        phipf_internal=phipf_internal,
        chipf_internal=chipf_internal,
        chips_eff=chips_eff,
    )

    trig = getattr(static_vmec, "trig_vmec", None)
    if trig is None:
        trig = vmec_trig_tables(
            ntheta=int(cfg.ntheta),
            nzeta=int(cfg.nzeta),
            nfp=int(wout_like.nfp),
            mmax=int(wout_like.mpol) - 1,
            nmax=int(wout_like.ntor),
            lasym=bool(wout_like.lasym),
            dtype=jnp.asarray(state0.Rcos).dtype,
        )
    if not bool(wout_like.lasym):
        # For lasym=False keep Z-force intact in the first-step diagnostic.
        zero_m1 = False

    constraint_tcon0: float | None = None
    if bool(include_constraint_force):
        constraint_tcon0 = float(indata.get_float("TCON0", 1.0))
    apply_lforbal = bool(indata.get_bool("LFORBAL", False)) if indata is not None else False

    def _apply_radial_tridi(rhs, alpha: float):
        return _radial_tridi_smooth_dirichlet(rhs, alpha=alpha, skip_nonpositive=True)

    def _metric_surface_precond_from_bcovar(bc):
        """Approximate radial preconditioner scaling from bcovar metrics.

        Only called from the host (never inside lax.scan traced code), so
        plain NumPy is sufficient and avoids JAX dispatch overhead.
        """
        guu = np.asarray(bc.guu)
        r12 = np.asarray(bc.jac.r12)
        bsubu = np.asarray(bc.bsubu)
        bsubv = np.asarray(bc.bsubv)
        nzeta = int(guu.shape[2])
        w_ang = np.asarray(vmec_wint_from_trig(trig, nzeta=nzeta)).astype(guu.dtype)
        return _metric_surface_precond_scales_np(guu=guu, r12=r12, bsubu=bsubu, bsubv=bsubv, w_ang=w_ang)

    def _pshalf_from_s(s_arr):
        return _pshalf_from_s_np(s_arr)

    def _sm_sp_from_s(s_arr):
        return _sm_sp_from_s_np(s_arr)

    def _lambda_preconditioner(bc, *, return_faclam: bool = False):
        from .preconditioner_1d_jax import lambda_preconditioner

        return lambda_preconditioner(
            bc=bc,
            trig=trig,
            s=s,
            cfg=cfg,
            return_faclam=return_faclam,
        )

    def _rz_preconditioner(frzl_in: TomnspsRZL, bc, k):
        from .preconditioner_1d_jax import rz_preconditioner

        return rz_preconditioner(
            frzl_in=frzl_in,
            bc=bc,
            k=k,
            trig=trig,
            s=s,
            cfg=cfg,
        )

    mask_pack = getattr(static_vmec, "tomnsps_masks", None)

    def _compute_forces(state: VMECState):
        k = vmec_forces_rz_from_wout(
            state=state,
            static=static_vmec,
            wout=wout_like,
            indata=None,
            constraint_tcon0=constraint_tcon0,
            use_vmec_synthesis=True,
            trig=trig,
        )
        frzl = vmec_residual_internal_from_kernels(
            k,
            cfg_ntheta=int(cfg.ntheta),
            cfg_nzeta=int(cfg.nzeta),
            wout=wout_like,
            trig=trig,
            apply_lforbal=apply_lforbal,
            include_edge=False,
            masks=mask_pack,
        )
        frzl = vmec_apply_scalxc_to_tomnsps(frzl=frzl, s=s)
        frzl_raw = frzl
        if bool(apply_m1_constraints):
            frzl = vmec_apply_m1_constraints(
                frzl=frzl,
                lconm1=bool(getattr(cfg, "lconm1", True)),
            )
        frzl = vmec_zero_m1_zforce(frzl=frzl, enabled=jnp.asarray(float(bool(zero_m1))))
        frzl = _zero_edge_rz_force_blocks(frzl, preserve_numpy=False)
        gcr2, gcz2, gcl2 = vmec_gcx2_from_tomnsps(
            frzl=frzl,
            lconm1=bool(getattr(cfg, "lconm1", True)),
            apply_m1_constraints=False,
            include_edge=bool(include_edge),
            apply_scalxc=False,
            s=s,
        )
        gcr2_raw, gcz2_raw, gcl2_raw = vmec_gcx2_from_tomnsps(
            frzl=frzl_raw,
            lconm1=bool(getattr(cfg, "lconm1", True)),
            apply_m1_constraints=False,
            include_edge=bool(include_edge),
            apply_scalxc=False,
            s=s,
        )
        norms = vmec_force_norms_from_bcovar_dynamic(bc=k.bc, trig=trig, s=s, signgs=signgs)
        fsqr = norms.r1 * norms.fnorm * gcr2
        fsqz = norms.r1 * norms.fnorm * gcz2
        fsql = norms.fnormL * gcl2
        rz_scale, l_scale = _metric_surface_precond_from_bcovar(k.bc)
        return k, frzl, fsqr, fsqz, fsql, rz_scale, l_scale, (gcr2_raw, gcz2_raw, gcl2_raw)

    k, frzl, fsqr, fsqz, fsql, rz_scale, l_scale, g_raw = _compute_forces(state0)
    gcr2_raw, gcz2_raw, gcl2_raw = g_raw

    if bool(use_axisymmetric_preconditioner) and (not bool(cfg.lthreed)) and (not bool(cfg.lasym)):
        lam_prec = _lambda_preconditioner(k.bc)
        frzl_pre = _rz_preconditioner(frzl, k.bc, k)
        frcc = jnp.asarray(frzl_pre.frcc)
        frss = frzl_pre.frss
        fzsc = jnp.asarray(frzl_pre.fzsc)
        fzcs = frzl_pre.fzcs
        flsc = jnp.asarray(frzl_pre.flsc) * jnp.asarray(lam_prec)
        flcs = frzl_pre.flcs
        if not (jnp.all(jnp.isfinite(frcc)) and jnp.all(jnp.isfinite(fzsc)) and jnp.all(jnp.isfinite(flsc))):
            frcc = jnp.asarray(frzl.frcc)
            frss = frzl.frss
            fzsc = jnp.asarray(frzl.fzsc)
            fzcs = frzl.fzcs
            flsc = jnp.asarray(frzl.flsc)
            flcs = frzl.flcs
    else:
        frcc = _apply_radial_tridi(frzl.frcc * rz_scale[:, None, None], precond_radial_alpha)
        frss = (
            _apply_radial_tridi(frzl.frss * rz_scale[:, None, None], precond_radial_alpha)
            if frzl.frss is not None
            else None
        )
        fzsc = _apply_radial_tridi(frzl.fzsc * rz_scale[:, None, None], precond_radial_alpha)
        fzcs = (
            _apply_radial_tridi(frzl.fzcs * rz_scale[:, None, None], precond_radial_alpha)
            if frzl.fzcs is not None
            else None
        )
        flsc = _apply_radial_tridi(frzl.flsc * l_scale[:, None, None], precond_lambda_alpha)
        flcs = (
            _apply_radial_tridi(frzl.flcs * l_scale[:, None, None], precond_lambda_alpha)
            if frzl.flcs is not None
            else None
        )

    frzl_pre = TomnspsRZL(
        frcc=frcc,
        frss=frss,
        fzsc=fzsc,
        fzcs=fzcs,
        flsc=flsc,
        flcs=flcs,
        frsc=getattr(frzl, "frsc", None),
        frcs=getattr(frzl, "frcs", None),
        fzcc=getattr(frzl, "fzcc", None),
        fzss=getattr(frzl, "fzss", None),
        flcc=getattr(frzl, "flcc", None),
        flss=getattr(frzl, "flss", None),
    )

    mpol = int(cfg.mpol)
    ntor = int(cfg.ntor)
    nrange = ntor + 1
    nfp = float(cfg.nfp)
    w_mode_mn = (1.0 + (jnp.arange(mpol)[:, None] ** 2 + (jnp.arange(nrange)[None, :] * nfp) ** 2)) ** (
        -float(mode_diag_exponent)
    )
    frcc_u = frcc * w_mode_mn[None, :, :]
    frss_u = (frss if frss is not None else jnp.zeros_like(frcc_u)) * w_mode_mn[None, :, :]
    fzsc_u = fzsc * w_mode_mn[None, :, :]
    fzcs_u = (fzcs if fzcs is not None else jnp.zeros_like(fzsc_u)) * w_mode_mn[None, :, :]
    flsc_u = flsc * w_mode_mn[None, :, :]
    flcs_u = (flcs if flcs is not None else jnp.zeros_like(flsc_u)) * w_mode_mn[None, :, :]

    def _mode_rms(a):
        a = jnp.asarray(a)
        return jnp.sqrt(jnp.mean(a * a, axis=0))

    frcc_mode = _mode_rms(frcc_u)
    fzsc_mode = _mode_rms(fzsc_u)
    flsc_mode = _mode_rms(flsc_u)

    gcr2_p, gcz2_p, gcl2_p = vmec_gcx2_from_tomnsps(
        frzl=frzl_pre,
        lconm1=bool(getattr(static.cfg, "lconm1", True)),
        apply_m1_constraints=False,
        include_edge=True,
        apply_scalxc=False,
        s=s,
    )
    rz_norm = vmec_rz_norm_from_state(
        state=state0,
        static=static,
        s=s,
        apply_scalxc=False,
        ns_min=0,
        ns_max=int(jnp.asarray(s).shape[0]),
    )
    norms = vmec_force_norms_from_bcovar_dynamic(bc=k.bc, trig=trig, s=s, signgs=signgs)
    f_norm1 = jnp.where(rz_norm != 0.0, 1.0 / rz_norm, jnp.asarray(float("inf"), dtype=rz_norm.dtype))
    delta_s = jnp.asarray(s[1] - s[0], dtype=rz_norm.dtype)
    fsqr1 = gcr2_p * f_norm1
    fsqz1 = gcz2_p * f_norm1
    fsql1 = gcl2_p * delta_s

    if step_size is None:
        time_step = float(indata.get_float("DELT", 5e-3))
    else:
        time_step = float(step_size)
    invtau = 0.15 / time_step
    otav = invtau
    dtau = time_step * otav / 2.0
    b1 = 1.0 - dtau
    fac = 1.0 / (1.0 + dtau)

    vRcc = fac * time_step * frcc_u
    vRss = fac * time_step * frss_u
    vZsc = fac * time_step * fzsc_u
    vZcs = fac * time_step * fzcs_u
    vLsc = fac * time_step * flsc_u
    vLcs = fac * time_step * flcs_u

    dRcc = time_step * vRcc
    dRss = time_step * vRss
    dZsc = time_step * vZsc
    dZcs = time_step * vZcs
    dLsc = time_step * vLsc
    dLcs = time_step * vLcs

    return {
        "fsqr": float(np.asarray(fsqr)),
        "fsqz": float(np.asarray(fsqz)),
        "fsql": float(np.asarray(fsql)),
        "fsqr1": float(np.asarray(fsqr1)),
        "fsqz1": float(np.asarray(fsqz1)),
        "fsql1": float(np.asarray(fsql1)),
        "gcr2_raw": float(np.asarray(gcr2_raw)),
        "gcz2_raw": float(np.asarray(gcz2_raw)),
        "gcl2_raw": float(np.asarray(gcl2_raw)),
        "rz_norm": float(np.asarray(rz_norm)),
        "f_norm1": float(np.asarray(f_norm1)),
        "f_norm_rz": float(np.asarray(norms.fnorm)),
        "f_norm_l": float(np.asarray(norms.fnormL)),
        "scalxc": np.asarray(vmec_scalxc_from_s(s=s, mpol=int(cfg.mpol))),
        "time_step": float(time_step),
        "dtau": float(dtau),
        "b1": float(b1),
        "fac": float(fac),
        "rz_scale": np.asarray(rz_scale),
        "l_scale": np.asarray(l_scale),
        "frzl": frzl,
        "frzl_pre": frzl_pre,
        "frcc_u": np.asarray(frcc_u),
        "frss_u": np.asarray(frss_u),
        "fzsc_u": np.asarray(fzsc_u),
        "fzcs_u": np.asarray(fzcs_u),
        "flsc_u": np.asarray(flsc_u),
        "flcs_u": np.asarray(flcs_u),
        "frcc_mode_rms": np.asarray(frcc_mode),
        "fzsc_mode_rms": np.asarray(fzsc_mode),
        "flsc_mode_rms": np.asarray(flsc_mode),
        "dRcc": np.asarray(dRcc),
        "dRss": np.asarray(dRss),
        "dZsc": np.asarray(dZsc),
        "dZcs": np.asarray(dZcs),
        "dLsc": np.asarray(dLsc),
        "dLcs": np.asarray(dLcs),
        "bcovar": k.bc,
    }
    if use_scan and dumps_enabled:
        raise ValueError("use_scan is incompatible with debug dumps (VMEC_JAX_DUMP_*).")
