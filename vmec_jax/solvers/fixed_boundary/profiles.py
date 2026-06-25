"""Pure profile and flux helper functions shared by solver entry points."""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from ..._compat import jnp
from ...field import TWOPI, chips_from_wout_chipf


class WoutLikeProfileSetup(NamedTuple):
    """Profiles and minimal WOUT-like metadata consumed by residual kernels."""

    flux: Any
    chipf_wout: Any
    phips: Any
    mass: Any
    pres: Any
    ncurr: int
    icurv: Any
    phipf_internal: Any
    chipf_internal: Any
    chips_eff: Any
    wout_like: Any


def _vmec_force_flux_profiles(*, phipf, chipf, signgs: int, flux_is_internal: bool, iotaf=None, iotas=None):
    phipf = jnp.asarray(phipf)
    chipf = None if chipf is None else jnp.asarray(chipf)
    if flux_is_internal:
        phipf_internal = phipf
        chipf_internal = chipf
    else:
        scale = jnp.asarray(TWOPI, dtype=phipf.dtype) * jnp.asarray(int(signgs), dtype=phipf.dtype)
        phipf_internal = phipf / scale
        chipf_internal = None if chipf is None else (chipf / scale)
    if chipf_internal is not None:
        chips_eff = chips_from_wout_chipf(
            chipf=chipf_internal,
            phipf=phipf_internal,
            iotaf=iotaf,
            iotas=iotas,
            assume_half_if_unknown=True,
        )
    else:
        iota = iotaf if iotaf is not None else iotas
        if iota is None:
            chips_eff = jnp.zeros_like(phipf_internal)
        else:
            chips_eff = jnp.asarray(iota, dtype=phipf_internal.dtype) * phipf_internal
    return phipf_internal, chipf_internal, chips_eff


def _s_half_from_full_mesh_s(s):
    s = jnp.asarray(s)
    if int(s.shape[0]) < 2:
        return s
    return jnp.concatenate([s[:1], 0.5 * (s[1:] + s[:-1])], axis=0)


def _half_mesh_from_full_mesh(x):
    x = jnp.asarray(x)
    if int(x.shape[0]) < 2:
        return x
    return jnp.concatenate([x[:1], 0.5 * (x[1:] + x[:-1])], axis=0)


def _pressure_half_mesh_from_indata(*, indata, s_full):
    from ...profiles import eval_profiles

    s_half = _s_half_from_full_mesh_s(s_full)
    prof = eval_profiles(indata, s_half)
    return jnp.asarray(prof.get("pressure", jnp.zeros_like(s_half)))


def _mass_half_mesh_from_indata(*, indata, s_full, phips, r00, gamma, lrfp: bool = False, chips=None):
    """Compute VMEC mass profile on half mesh: mass = pmass * (|vnorm|*r00)^gamma."""
    from ...profiles import eval_profiles

    s_half = _s_half_from_full_mesh_s(s_full)
    prof = eval_profiles(indata, s_half)
    pmass = jnp.asarray(prof.get("pressure", jnp.zeros_like(s_half)))
    vnorm = jnp.asarray(phips)
    if lrfp and (chips is not None):
        vnorm = jnp.asarray(chips)
    mass = pmass * (jnp.abs(vnorm) * jnp.asarray(r00, dtype=pmass.dtype)) ** jnp.asarray(gamma, dtype=pmass.dtype)
    if int(mass.shape[0]) > 0:
        mass = mass.at[0].set(jnp.asarray(0.0, dtype=mass.dtype))
    return mass


def _icurv_full_mesh_from_indata(*, indata, s_full, signgs: int):
    from ...profiles import eval_profiles

    s_full = jnp.asarray(s_full)
    ncurr = int(indata.get_int("NCURR", 0))
    if ncurr != 1:
        return jnp.zeros_like(s_full)

    curtor = float(indata.get_float("CURTOR", 0.0))
    if abs(curtor) <= np.finfo(float).eps:
        return jnp.zeros_like(s_full)

    # VMEC stores icurv on the half mesh (same indexing as phips/chips/iotas),
    # evaluated at s = (i-1.5)*hs for i>=2. Mirror that here.
    s_half = _s_half_from_full_mesh_s(s_full)
    prof = eval_profiles(indata, s_half)
    icurv_raw = jnp.asarray(prof.get("current", jnp.zeros_like(s_half)))
    if int(icurv_raw.shape[0]) != int(s_full.shape[0]):
        icurv_raw = jnp.zeros_like(s_half)

    # VMEC scales by pcurr(1) (edge), not the last half-mesh value.
    pedge_prof = eval_profiles(indata, jnp.asarray([1.0], dtype=s_full.dtype))
    pedge = jnp.asarray(pedge_prof.get("current", jnp.asarray([0.0], dtype=s_full.dtype)))[0]
    valid_pedge = jnp.abs(pedge) > jnp.asarray(abs(np.finfo(float).eps * curtor), dtype=s_full.dtype)

    mu0 = 4e-7 * np.pi
    currv = mu0 * curtor
    denom = jnp.where(valid_pedge, pedge, jnp.asarray(1.0, dtype=s_full.dtype))
    scale = jnp.asarray(float(signgs) * currv / (2.0 * np.pi), dtype=icurv_raw.dtype) / denom
    icurv = jnp.where(valid_pedge, scale * icurv_raw, jnp.zeros_like(s_full))
    if int(icurv.shape[0]) > 0:
        icurv = icurv.at[0].set(0.0)
    return icurv


def build_wout_like_profiles_from_indata(
    *,
    indata,
    static,
    s_profile,
    signgs: int,
    idx00: int,
    prefer_host_default_profiles: bool,
    s_profile_has_tracer: bool,
) -> WoutLikeProfileSetup:
    """Build VMEC force profiles and the minimal WOUT-like force container.

    Residual kernels need a compact WOUT-like object instead of the full file
    output structure.  This setup step is deterministic for a given input deck,
    static grid, and radial mesh, so it belongs outside the residual iteration
    loop and can be tested independently from the nonlinear controller.
    """

    from ...boundary import boundary_from_indata
    from ...energy import flux_profiles_from_indata, flux_profiles_from_indata_host_default
    from .results import WoutLikeVmecForces

    flux = None
    if bool(prefer_host_default_profiles) and not bool(s_profile_has_tracer):
        try:
            flux = flux_profiles_from_indata_host_default(indata, s_profile, signgs=signgs)
        except Exception:
            flux = None
    if flux is None:
        flux = flux_profiles_from_indata(indata, s_profile, signgs=signgs)
    chipf_wout = jnp.asarray(flux.chipf)

    phips = jnp.asarray(flux.phips)
    if phips.shape[0] >= 1:
        phips = phips.at[0].set(0.0)

    boundary = boundary_from_indata(indata, static.modes)
    r00 = (
        float(np.asarray(boundary.R_cos)[int(idx00)])
        if int(idx00) >= 0
        else float(np.asarray(boundary.R_cos)[0])
    )
    gamma = float(indata.get_float("GAMMA", 0.0))
    lrfp = bool(indata.get_bool("LRFP", False))
    chips = _half_mesh_from_full_mesh(chipf_wout) if lrfp else None
    mass = _mass_half_mesh_from_indata(
        indata=indata,
        s_full=s_profile,
        phips=phips,
        r00=r00,
        gamma=gamma,
        lrfp=lrfp,
        chips=chips,
    )

    pres = _pressure_half_mesh_from_indata(indata=indata, s_full=s_profile)
    ncurr = int(indata.get_int("NCURR", 0))
    icurv = _icurv_full_mesh_from_indata(indata=indata, s_full=s_profile, signgs=signgs)
    phipf_internal, chipf_internal, chips_eff = _vmec_force_flux_profiles(
        phipf=jnp.asarray(flux.phipf),
        chipf=chipf_wout,
        signgs=signgs,
        flux_is_internal=True,
    )

    wout_like = WoutLikeVmecForces(
        nfp=int(static.cfg.nfp),
        mpol=int(static.cfg.mpol),
        ntor=int(static.cfg.ntor),
        lasym=bool(static.cfg.lasym),
        signgs=int(signgs),
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
    return WoutLikeProfileSetup(
        flux=flux,
        chipf_wout=chipf_wout,
        phips=phips,
        mass=mass,
        pres=pres,
        ncurr=ncurr,
        icurv=icurv,
        phipf_internal=phipf_internal,
        chipf_internal=chipf_internal,
        chips_eff=chips_eff,
        wout_like=wout_like,
    )
