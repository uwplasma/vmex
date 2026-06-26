"""Post-solve flux/profile reconciliation helpers for the VMEC driver."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import numpy as np

from ..boundary import boundary_from_indata
from ..energy import FluxProfiles, _iotaf_from_iotas


def profiles_from_static(
    *,
    indata,
    static_in,
    signgs: int,
    flux_profiles_from_indata_host_default_func,
    flux_profiles_from_indata_func,
    eval_profiles_func,
) -> tuple[Any, dict, Any]:
    """Build flux, VMEC profile, and pressure arrays for one static grid."""

    flux_local = flux_profiles_from_indata_host_default_func(indata, static_in.s, signgs=int(signgs))
    if flux_local is None:
        flux_local = flux_profiles_from_indata_func(indata, static_in.s, signgs=int(signgs))
    # VMEC evaluates pressure/iota/current profiles on the radial half mesh.
    if int(static_in.cfg.ns) < 2:
        s_half = np.asarray(static_in.s)
    else:
        s_full = np.asarray(static_in.s)
        s_half = np.concatenate([s_full[:1], 0.5 * (s_full[1:] + s_full[:-1])], axis=0)
    prof_local = eval_profiles_func(indata, s_half)
    pressure_local = prof_local.get("pressure", np.zeros_like(np.asarray(static_in.s)))
    return flux_local, prof_local, pressure_local


def finalize_flux_profiles_for_run(
    *,
    cfg,
    indata,
    static,
    result,
    signgs: int,
    flux_local,
    profiles_local,
    pressure_local,
    static_profile_cache,
    final_flux_profiles_from_state_func,
) -> tuple[Any, Any, dict]:
    """Return final static, flux, and profiles for a completed driver result."""

    if flux_local is None or profiles_local is None or pressure_local is None:
        if static is None:
            static = static_profile_cache.build_static_cfg(cfg)
        flux_local, profiles_local, pressure_local = static_profile_cache.profiles_for_static(static)
    flux_out, profiles_out = final_flux_profiles_from_state_func(
        indata=indata,
        static_in=static,
        state=result.state,
        signgs=signgs,
        flux_local=flux_local,
        prof_local=profiles_local,
        pressure_local=pressure_local,
    )
    return static, flux_out, profiles_out


def final_flux_profiles_from_state(
    *,
    indata,
    static_in,
    state,
    signgs: int,
    flux_local,
    prof_local: dict,
    pressure_local,
    boundary_from_indata_func=boundary_from_indata,
    iotaf_from_iotas_func=_iotaf_from_iotas,
):
    """Return post-solve flux/profile payloads consistent with the solved state.

    ``flux_profiles_from_indata()`` is input-only. For current-driven runs
    (``NCURR=1``), VMEC updates the rotational transform from the solved force
    balance. Mirror that here so the driver returns the same effective flux/iota
    channels that ``wout`` reconstruction uses.
    """

    ncurr = int(indata.get_int("NCURR", 0))
    if ncurr != 1:
        return flux_local, prof_local
    if os.getenv("VMEC_JAX_DISABLE_WOUT_NCURR_RECOMPUTE", "0") not in ("", "0"):
        return flux_local, prof_local

    from ..kernels.bcovar import vmec_bcovar_half_mesh_from_wout
    from ..kernels.residue import vmec_pwint_from_trig
    from ..kernels.tomnsp import vmec_trig_tables
    from ..wout import _chipf_from_chips, _icurv_full_mesh_from_indata

    traced = False
    try:
        import jax

        traced = any(
            isinstance(x, jax.core.Tracer)
            for x in (
                getattr(state, "Rcos", None),
                getattr(state, "Rsin", None),
                getattr(state, "Zcos", None),
                getattr(state, "Zsin", None),
                getattr(state, "Lcos", None),
                getattr(state, "Lsin", None),
                getattr(flux_local, "phipf", None),
                getattr(flux_local, "phips", None),
                getattr(flux_local, "chipf", None),
                getattr(flux_local, "lamscale", None),
                pressure_local,
            )
        )
    except Exception:
        traced = False

    xp = jax.numpy if traced else np

    def _asarray(x):
        return xp.asarray(x, dtype=float)

    def _set_axis_zero(arr):
        if int(arr.shape[0]) == 0:
            return arr
        if traced:
            return arr.at[0].set(0.0)
        arr = arr.copy()
        arr[0] = 0.0
        return arr

    s = _asarray(static_in.s)
    try:
        state_ns = int(getattr(state.Rcos, "shape", _asarray(state.Rcos).shape)[0])
    except Exception:
        state_ns = int(s.shape[0])
    if int(state_ns) != int(s.shape[0]):
        return flux_local, prof_local
    phipf = _asarray(flux_local.phipf)
    phips = _set_axis_zero(_asarray(flux_local.phips))

    pressure_out = _set_axis_zero(_asarray(pressure_local))

    gamma = float(indata.get_float("GAMMA", 0.0))
    lrfp = bool(indata.get_bool("LRFP", False))
    boundary = boundary_from_indata_func(indata, static_in.modes)
    idx00 = np.where((np.asarray(static_in.modes.m) == 0) & (np.asarray(static_in.modes.n) == 0))[0]
    r00 = float(boundary.R_cos[int(idx00[0])]) if idx00.size else float(np.asarray(boundary.R_cos)[0])
    vnorm = phips
    if lrfp:
        chipf_in = _asarray(flux_local.chipf)
        if int(chipf_in.shape[0]) > 0:
            chips_in = xp.concatenate([chipf_in[:1], 0.5 * (chipf_in[1:] + chipf_in[:-1])], axis=0)
            vnorm = chips_in
    mass = pressure_out * (xp.abs(vnorm) * r00) ** gamma
    mass = _set_axis_zero(mass)

    trig = getattr(static_in, "trig_vmec", None)
    if trig is None:
        trig = vmec_trig_tables(
            ntheta=int(np.asarray(static_in.grid.theta).shape[0]),
            nzeta=int(np.asarray(static_in.grid.zeta).shape[0]),
            nfp=int(static_in.cfg.nfp),
            mmax=max(0, int(static_in.cfg.mpol) - 1),
            nmax=max(0, int(static_in.cfg.ntor)),
            lasym=bool(static_in.cfg.lasym),
        )
    icurv = _asarray(_icurv_full_mesh_from_indata(indata=indata, s_full=s, signgs=int(signgs)))
    wout_like_pre = SimpleNamespace(
        phipf=phipf,
        phips=phips,
        chipf=xp.zeros_like(phipf),
        signgs=int(signgs),
        nfp=int(static_in.cfg.nfp),
        mpol=int(static_in.cfg.mpol),
        ntor=int(static_in.cfg.ntor),
        lasym=bool(static_in.cfg.lasym),
        ncurr=0,
        lcurrent=False,
        icurv=xp.zeros_like(phipf),
        flux_is_internal=True,
        mass=mass,
        gamma=gamma,
    )
    if traced:
        bc_pre = vmec_bcovar_half_mesh_from_wout(
            state=state,
            static=static_in,
            wout=wout_like_pre,
            pres=pressure_out,
            use_vmec_synthesis=True,
            trig=trig,
        )
    else:
        from ..kernels.numpy_forces import _numpy_module_patch

        with _numpy_module_patch():
            bc_pre = vmec_bcovar_half_mesh_from_wout(
                state=state,
                static=static_in,
                wout=wout_like_pre,
                pres=pressure_out,
                use_vmec_synthesis=True,
                trig=trig,
            )

    sqrtg = _asarray(bc_pre.jac.sqrtg)
    safe_sqrtg = xp.where(sqrtg != 0.0, sqrtg, 1.0)
    overg = xp.where(sqrtg != 0.0, 1.0 / safe_sqrtg, 0.0)
    pwint = _asarray(
        vmec_pwint_from_trig(trig, ns=int(overg.shape[0]), nzeta=int(overg.shape[2])),
    )
    guu = _asarray(bc_pre.guu)
    guv = _asarray(bc_pre.guv)
    bsupu = _asarray(bc_pre.bsupu)
    bsupv = _asarray(bc_pre.bsupv)
    top = icurv - xp.sum(pwint * ((guu * bsupu) + (guv * bsupv)), axis=(1, 2))
    bot = xp.sum(pwint * (overg * guu), axis=(1, 2))
    safe_bot = xp.where(bot != 0.0, bot, 1.0)
    chips = xp.where(bot != 0.0, top / safe_bot, 0.0)
    chips = _set_axis_zero(chips)

    safe_phips = xp.where(phips != 0.0, phips, 1.0)
    iotas = xp.where(phips != 0.0, chips / safe_phips, 0.0)
    iotas = _set_axis_zero(iotas)
    iotaf = _asarray(iotaf_from_iotas_func(iotas, lrfp=lrfp))
    chipf = _asarray(_chipf_from_chips(chips))

    prof_out = dict(prof_local)
    prof_out["iota"] = iotas
    prof_out["iotaf"] = iotaf
    flux_out = FluxProfiles(
        phipf=phipf,
        chipf=chipf,
        phips=phips,
        signgs=int(signgs),
        lamscale=_asarray(flux_local.lamscale),
    )
    return flux_out, prof_out
