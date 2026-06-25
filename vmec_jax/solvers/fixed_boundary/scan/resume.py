"""Scan-resume initialization helpers for VMEC2000-style residual iteration."""

from __future__ import annotations

from typing import Any, NamedTuple

from ...._compat import jnp
from ..results import ScanCarry


class ScanResumeInitialFields(NamedTuple):
    """Initial carry fields restored from an optional residual-iteration resume state."""

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


def initialize_scan_resume_state(
    resume_state: dict | None,
    *,
    dtype: Any,
    velocity_shape: tuple[int, ...],
    k_ndamp: int,
    time_step_default: Any,
    flip_sign_default: Any,
    state_checkpoint_default: Any,
) -> ScanResumeInitialFields:
    """Build initial scan carry values from defaults plus an optional resume payload."""

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

    return ScanResumeInitialFields(
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


def build_initial_scan_carry(
    *,
    state_init: Any,
    resume_fields: ScanResumeInitialFields,
    dtype: Any,
    iter_offset0: int,
    cache_valid: Any,
    cache_precond_diag: Any,
    cache_tcon: Any,
    cache_norms: Any,
    cache_rz_scale: Any,
    cache_l_scale: Any,
    cache_rz_norm: Any,
    cache_f_norm1: Any,
    cache_rz_mats: Any,
    cache_lam_prec: Any,
    edge_Rcos: Any,
    edge_Rsin: Any,
    edge_Zcos: Any,
    edge_Zsin: Any,
) -> ScanCarry:
    """Build the initial VMEC2000 scan carry from resume and cache fields."""

    return ScanCarry(
        state=state_init,
        time_step=resume_fields.time_step,
        inv_tau=resume_fields.inv_tau,
        fsq_prev=resume_fields.fsq_prev,
        fsq0_prev=resume_fields.fsq0_prev,
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
        vRcc=resume_fields.vRcc,
        vRss=resume_fields.vRss,
        vZsc=resume_fields.vZsc,
        vZcs=resume_fields.vZcs,
        vLsc=resume_fields.vLsc,
        vLcs=resume_fields.vLcs,
        vRsc=resume_fields.vRsc,
        vRcs=resume_fields.vRcs,
        vZcc=resume_fields.vZcc,
        vZss=resume_fields.vZss,
        vLcc=resume_fields.vLcc,
        vLss=resume_fields.vLss,
        flip_sign=resume_fields.flip_sign,
        iter_offset=jnp.asarray(iter_offset0, dtype=jnp.int32),
        iter1=resume_fields.iter1,
        res0=resume_fields.res0,
        res1=resume_fields.res1,
        state_checkpoint=resume_fields.state_checkpoint,
        cache_valid=cache_valid,
        cache_precond_diag=cache_precond_diag,
        cache_tcon=cache_tcon,
        cache_norms=cache_norms,
        cache_rz_scale=cache_rz_scale,
        cache_l_scale=cache_l_scale,
        cache_rz_norm=cache_rz_norm,
        cache_f_norm1=cache_f_norm1,
        cache_prec_rz_mats=cache_rz_mats,
        cache_prec_lam_prec=cache_lam_prec,
        force_bcovar_update=resume_fields.force_bcovar_update,
        ijacob=resume_fields.ijacob,
        bad_resets=resume_fields.bad_resets,
        bad_growth=resume_fields.bad_growth,
        fsqz_prev=resume_fields.fsqz_prev,
        r00_prev=resume_fields.r00_prev,
        z00_prev=resume_fields.z00_prev,
        w_mhd_prev=resume_fields.w_mhd_prev,
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


def build_traced_scan_resume_state(carry_final: Any, *, max_iter: int) -> dict[str, Any]:
    """Return the differentiable resume payload for a traced scan result.

    This mirrors the full host resume payload but keeps all values as JAX
    arrays so callers can stage or differentiate scan solves without forcing
    host conversion.
    """
    return {
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
