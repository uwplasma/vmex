"""Scan-resume initialization helpers for VMEC2000-style residual iteration."""

from __future__ import annotations

from typing import Any, NamedTuple

from ._compat import jnp


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
