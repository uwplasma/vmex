"""Minimal `wout_*.nc` reader helpers.

This module is intentionally small and only depends on `netCDF4` when used.
It is meant for regression comparisons against VMEC2000 outputs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from ._compat import has_jax, jax, jnp
from .modes import vmec_mode_table
from .modes import nyquist_mode_table_from_grid
from .namelist import InData
from .state import StateLayout, VMECState
from .fourier import eval_fourier
from .vmec_parity import (
    vmec_m1_internal_to_physical_signed,
    vmec_m1_internal_to_physical_signed_host,
    vmec_m1_physical_to_internal_signed,
)
from .vmec_realspace import (
    vmec_realspace_synthesis,
    vmec_realspace_synthesis_dtheta,
    vmec_realspace_geom_from_state,
)
from .vmec_residue import vmec_pwint_from_trig
from .io.wout import bsubs as _wout_bsubs_helpers
from .io.wout import diagnostics as _wout_diagnostics
from .io.wout import flux as _wout_flux_helpers
from .io.wout import jxbforce as _wout_jxbforce_helpers
from .io.wout import mercier as _wout_mercier
from .io.wout import parity as _wout_parity_helpers
from .io.wout.netcdf import (
    read_mode_table,
    read_nyquist_fourier_fields,
    read_optional_int_scalar,
    read_type_field,
    read_wout_scalar_metadata,
    write_fixed_width_string_variable,
    write_float_variable,
    write_int_variable,
    write_nyquist_fourier_fields,
)
from .io.wout.nyquist import (
    apply_nyquist_half_weight as _apply_nyquist_half_weight,  # noqa: F401 - compatibility export
    vmec_jxbforce_cos_coeffs as _vmec_jxbforce_cos_coeffs,  # noqa: F401 - compatibility export
    vmec_jxbforce_sin_coeffs as _vmec_jxbforce_sin_coeffs,  # noqa: F401 - compatibility export
    vmec_symforce_antisym as _vmec_symforce_antisym,  # noqa: F401 - compatibility export
    vmec_symforce_apply as _vmec_symforce_apply,
    vmec_symoutput_expand as _vmec_symoutput_expand,
    vmec_symoutput_split as _vmec_symoutput_split,
    vmec_wrout_lasym_bsubuv_output_scale as _vmec_wrout_lasym_bsubuv_output_scale,
    vmec_wrout_nyquist_cos_coeffs as _vmec_wrout_nyquist_cos_coeffs,
    vmec_wrout_nyquist_lasym_loop as _vmec_wrout_nyquist_lasym_loop,
    vmec_wrout_nyquist_sin_coeffs as _vmec_wrout_nyquist_sin_coeffs,
    vmec_wrout_nyquist_sin_coeffs_loop as _vmec_wrout_nyquist_sin_coeffs_loop,
    vmec_wrout_nyquist_synthesis as _vmec_wrout_nyquist_synthesis,  # noqa: F401 - compatibility export
)
from .io.wout.schema import WoutData, _bool_from_nc, _nc_scalar, assert_main_modes_match_wout
from .vmec_tomnsp import vmec_trig_tables


MU0 = 4e-7 * np.pi  # N/A^2
_chipf_from_chips = _wout_flux_helpers.chipf_from_chips
_compute_aspectratio = _wout_diagnostics.compute_aspectratio
_compute_ctor_from_buco = _wout_diagnostics.compute_ctor_from_buco
_compute_eqfor_beta = _wout_diagnostics.compute_eqfor_beta
_compute_eqfor_betaxis = _wout_diagnostics.compute_eqfor_betaxis
_glasser_from_wout_mercier_terms = _wout_diagnostics.glasser_from_wout_mercier_terms
_glasser_profiles_from_wout_data = _wout_diagnostics.glasser_profiles_from_wout_data
_glasser_profiles_from_wout_variables = _wout_diagnostics.glasser_profiles_from_wout_variables
_icurv_full_mesh_from_indata = _wout_flux_helpers.icurv_full_mesh_from_indata
_lambda_full_from_wout_half_mesh = _wout_flux_helpers.lambda_full_from_wout_half_mesh
_lambda_half_mesh_weights = _wout_diagnostics.lambda_half_mesh_weights
_lambda_wout_from_full_mesh = _wout_flux_helpers.lambda_wout_from_full_mesh
_pshalf_from_s = _wout_diagnostics.pshalf_from_s
_safe_divide = _wout_diagnostics.safe_divide
_wout_current_profile_metadata_from_indata = _wout_flux_helpers.wout_current_profile_metadata_from_indata
_wout_phi_profile_from_variables = _wout_flux_helpers.wout_phi_profile_from_variables
_read_wout_scalar_metadata = read_wout_scalar_metadata
_bss_scalxc_undo_factor = _wout_parity_helpers.bss_scalxc_undo_factor
_bss_should_undo_scalxc = _wout_parity_helpers.bss_should_undo_scalxc
_undo_bss_scalxc_if_enabled = _wout_parity_helpers.undo_bss_scalxc_if_enabled
_filter_bsubuv_jxbforce = _wout_jxbforce_helpers._filter_bsubuv_jxbforce
_filter_bsubuv_jxbforce_lasym_loop = _wout_jxbforce_helpers._filter_bsubuv_jxbforce_lasym_loop
_filter_bsubuv_jxbforce_loop = _wout_jxbforce_helpers._filter_bsubuv_jxbforce_loop
_filter_bsubuv_jxbforce_parity = _wout_jxbforce_helpers._filter_bsubuv_jxbforce_parity
_filter_bsubuv_jxbforce_parity_loop = _wout_jxbforce_helpers._filter_bsubuv_jxbforce_parity_loop
_jxbforce_bsubsu_bsubsv_loop = _wout_jxbforce_helpers._jxbforce_bsubsu_bsubsv_loop
_jxbforce_filter_with_bsubs_derivs_loop = _wout_jxbforce_helpers._jxbforce_filter_with_bsubs_derivs_loop
_jxbforce_nyquist_limits = _wout_jxbforce_helpers._jxbforce_nyquist_limits


def _vmec_wint_from_trig(trig) -> np.ndarray:
    """Return VMEC-style angular weights (wint) on the internal grid."""
    cosmui3 = np.asarray(trig.cosmui3)
    mscale = np.asarray(trig.mscale)
    if cosmui3.ndim != 2:
        raise ValueError("Expected trig.cosmui3 with shape (ntheta3, mmax+1)")
    if mscale.size == 0:
        raise ValueError("Expected non-empty trig.mscale")
    w_theta = cosmui3[:, 0] / float(mscale[0])
    nzeta = int(np.asarray(trig.cosnv).shape[0])
    wint = w_theta[:, None] * np.ones((nzeta,), dtype=w_theta.dtype)[None, :]
    return np.asarray(wint, dtype=float)


def _vmec_wint_from_trig_jax(trig):
    """Return VMEC-style angular weights on the internal grid as JAX arrays."""
    cosmui3 = jnp.asarray(trig.cosmui3)
    mscale = jnp.asarray(trig.mscale)
    if cosmui3.ndim != 2:
        raise ValueError("Expected trig.cosmui3 with shape (ntheta3, mmax+1)")
    if int(mscale.size) == 0:
        raise ValueError("Expected non-empty trig.mscale")
    w_theta = cosmui3[:, 0] / mscale[0]
    nzeta = int(np.asarray(trig.cosnv).shape[0])
    return w_theta[:, None] * jnp.ones((nzeta,), dtype=w_theta.dtype)[None, :]


def _bcovar_from_force_payload_with_geometry(geom: dict[str, Any], k_force) -> Any:
    """Attach VMEC force-kernel geometry channels and return the bcovar payload.

    ``wout_minimal_from_fixed_boundary`` can reuse force kernels produced at the
    final solve point or recompute them for output.  Both paths must populate
    the same VMEC ``pr*/pz*`` geometry channels before the Mercier/JXBFORCE
    reducers run, so keep that copy in one place.
    """

    geom["pr1_even"] = np.asarray(k_force.pr1_even, dtype=float)
    geom["pr1_odd"] = np.asarray(k_force.pr1_odd, dtype=float)
    geom["pz1_even"] = np.asarray(k_force.pz1_even, dtype=float)
    geom["pz1_odd"] = np.asarray(k_force.pz1_odd, dtype=float)
    geom["pru_even"] = np.asarray(k_force.pru_even, dtype=float)
    geom["pru_odd"] = np.asarray(k_force.pru_odd, dtype=float)
    geom["pzu_even"] = np.asarray(k_force.pzu_even, dtype=float)
    geom["pzu_odd"] = np.asarray(k_force.pzu_odd, dtype=float)
    geom["prv_even"] = np.asarray(k_force.prv_even, dtype=float)
    geom["prv_odd"] = np.asarray(k_force.prv_odd, dtype=float)
    geom["pzv_even"] = np.asarray(k_force.pzv_even, dtype=float)
    geom["pzv_odd"] = np.asarray(k_force.pzv_odd, dtype=float)
    return k_force.bc


def equilibrium_aspect_ratio_from_state(*, state: VMECState, static) -> Any:
    """Compute VMEC's equilibrium aspect ratio directly from a solved state.

    This mirrors the ``aspectratio.f`` path used during ``wout`` synthesis, but
    keeps the calculation on JAX arrays so callers can differentiate through the
    solved equilibrium without materializing a full ``wout`` object.
    """
    cfg = static.cfg
    trig = getattr(static, "trig_vmec", None)
    if trig is None:
        trig = vmec_trig_tables(
            ntheta=int(cfg.ntheta),
            nzeta=int(cfg.nzeta),
            nfp=int(cfg.nfp),
            mmax=int(cfg.mpol),
            nmax=int(cfg.ntor),
            lasym=bool(cfg.lasym),
            dtype=jnp.asarray(state.Rcos).dtype,
            cache=False,
        )
    geom = _vmec_realspace_geom_light_from_state(
        state=state,
        modes=static.modes,
        trig=trig,
        lasym=bool(cfg.lasym),
    )
    R = jnp.asarray(geom["R"])
    Zu = jnp.asarray(geom["Zu"])
    wint = _vmec_wint_from_trig_jax(trig)
    rb = R[-1]
    zub = Zu[-1]
    t1 = rb * zub * wint
    volume_p = (2.0 * jnp.pi * jnp.pi) * jnp.abs(jnp.sum(rb * t1))
    cross_area_p = (2.0 * jnp.pi) * jnp.abs(jnp.sum(t1))
    cross_area_safe = jnp.where(cross_area_p != 0.0, cross_area_p, 1.0)
    Aminor_p = jnp.where(cross_area_p != 0.0, jnp.sqrt(cross_area_safe / jnp.pi), 0.0)
    Rmajor_p = jnp.where(cross_area_p != 0.0, volume_p / (2.0 * jnp.pi * cross_area_safe), 0.0)
    return jnp.where(Aminor_p != 0.0, Rmajor_p / Aminor_p, 0.0)


def equilibrium_iota_profiles_from_state(*, state: VMECState, static, indata, signgs: int):
    """Compute VMEC-consistent current/iota profiles from a solved state.

    For ``NCURR=0`` this returns the prescribed input profile. For
    ``NCURR=1`` it mirrors VMEC's ``add_fluxes``-style recomputation from the
    solved force-balance state so callers can differentiate a current-driven
    iota target without reimplementing the wout synthesis logic.
    """
    from types import SimpleNamespace

    from .boundary import boundary_from_indata
    from .energy import _iotaf_from_iotas, flux_profiles_from_indata
    from .profiles import eval_profiles
    from .solvers.fixed_boundary.profiles import _half_mesh_from_full_mesh, _mass_half_mesh_from_indata
    from .vmec_bcovar import vmec_bcovar_half_mesh_from_wout

    s = jnp.asarray(static.s)
    if s.shape[0] < 2:
        s_half = s
    else:
        s_half = jnp.concatenate([s[:1], 0.5 * (s[1:] + s[:-1])], axis=0)

    flux = flux_profiles_from_indata(indata, s, signgs=int(signgs))
    phipf = jnp.asarray(flux.phipf)
    phips = jnp.asarray(flux.phips)
    if phips.shape[0] >= 1:
        phips = phips.at[0].set(0.0)

    prof = eval_profiles(indata, s_half)
    iotas = jnp.asarray(prof.get("iota", jnp.zeros_like(s_half)))
    if iotas.shape[0] >= 1:
        iotas = iotas.at[0].set(0.0)
    if int(indata.get_int("NCURR", 0)) == 0:
        chips = iotas * phips
        iotaf = jnp.asarray(_iotaf_from_iotas(iotas, lrfp=bool(indata.get_bool("LRFP", False))))
        return chips, iotas, iotaf

    boundary = boundary_from_indata(indata, static.modes)
    idx00 = np.where((np.asarray(static.modes.m) == 0) & (np.asarray(static.modes.n) == 0))[0]
    r00 = float(np.asarray(boundary.R_cos)[int(idx00[0])]) if idx00.size else float(np.asarray(boundary.R_cos)[0])
    gamma = float(indata.get_float("GAMMA", 0.0))
    lrfp = bool(indata.get_bool("LRFP", False))
    chips_half = _half_mesh_from_full_mesh(jnp.asarray(flux.chipf)) if lrfp else None
    mass = _mass_half_mesh_from_indata(
        indata=indata,
        s_full=s,
        phips=phips,
        r00=r00,
        gamma=gamma,
        lrfp=lrfp,
        chips=chips_half,
    )
    pres = jnp.asarray(prof.get("pressure", jnp.zeros_like(s_half)))
    if pres.shape[0] >= 1:
        pres = pres.at[0].set(0.0)
    icurv = jnp.asarray(_icurv_full_mesh_from_indata(indata=indata, s_full=s, signgs=int(signgs)))

    trig = getattr(static, "trig_vmec", None)
    if trig is None:
        trig = vmec_trig_tables(
            ntheta=int(static.cfg.ntheta),
            nzeta=int(static.cfg.nzeta),
            nfp=int(static.cfg.nfp),
            mmax=int(np.max(np.asarray(static.modes.m))),
            nmax=int(np.max(np.abs(np.asarray(static.modes.n)))),
            lasym=bool(static.cfg.lasym),
            dtype=jnp.asarray(state.Rcos).dtype,
        )

    wout_like_pre = SimpleNamespace(
        phipf=phipf,
        phips=phips,
        chipf=jnp.zeros_like(phipf),
        signgs=int(signgs),
        nfp=int(static.cfg.nfp),
        mpol=int(static.cfg.mpol),
        ntor=int(static.cfg.ntor),
        lasym=bool(static.cfg.lasym),
        ncurr=0,
        lcurrent=False,
        icurv=jnp.zeros_like(phipf),
        flux_is_internal=True,
        mass=mass,
        gamma=gamma,
    )
    bc_pre = vmec_bcovar_half_mesh_from_wout(
        state=state,
        static=static,
        wout=wout_like_pre,
        pres=pres,
        use_vmec_synthesis=True,
        trig=trig,
    )

    sqrtg = jnp.asarray(bc_pre.jac.sqrtg)
    sqrtg_safe = jnp.where(sqrtg != 0.0, sqrtg, jnp.ones_like(sqrtg))
    overg = jnp.where(sqrtg != 0.0, 1.0 / sqrtg_safe, 0.0)
    pwint = jnp.asarray(
        vmec_pwint_from_trig(trig, ns=int(overg.shape[0]), nzeta=int(overg.shape[2])),
        dtype=overg.dtype,
    )
    guu = jnp.asarray(bc_pre.guu)
    guv = jnp.asarray(bc_pre.guv)
    bsupu = jnp.asarray(bc_pre.bsupu)
    bsupv = jnp.asarray(bc_pre.bsupv)

    top = jnp.asarray(icurv, dtype=overg.dtype) - jnp.sum(
        pwint * ((guu * bsupu) + (guv * bsupv)),
        axis=(1, 2),
    )
    bot = jnp.sum(pwint * (overg * guu), axis=(1, 2))
    bot_safe = jnp.where(bot != 0.0, bot, jnp.ones_like(bot))
    chips = jnp.where(bot != 0.0, top / bot_safe, jnp.zeros_like(top))
    if chips.shape[0] >= 1:
        chips = chips.at[0].set(0.0)
    phips_safe = jnp.where(phips != 0.0, phips, jnp.ones_like(phips))
    iotas = jnp.where(phips != 0.0, chips / phips_safe, jnp.zeros_like(chips))
    if iotas.shape[0] >= 1:
        iotas = iotas.at[0].set(0.0)
    iotaf = jnp.asarray(_iotaf_from_iotas(iotas, lrfp=lrfp))
    return chips, iotas, iotaf


def _compute_equif_wout(
    *,
    bsubu: np.ndarray,
    bsubv: np.ndarray,
    pres: np.ndarray,
    vp: np.ndarray,
    phipf: np.ndarray,
    chipf: np.ndarray,
    signgs: int,
    trig,
    s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute buco/bvco/jcuru/jcurv/equif with VMEC eqfor normalization."""
    s = np.asarray(s, dtype=float)
    ns = int(s.shape[0])
    if ns < 3:
        z = np.zeros((ns,), dtype=float)
        return z.copy(), z.copy(), z.copy(), z.copy(), z.copy()

    hs = float(s[1] - s[0])
    ohs = 1.0 / hs if hs != 0.0 else 0.0
    wint = _vmec_wint_from_trig(trig)
    bsubu = np.asarray(bsubu, dtype=float)
    bsubv = np.asarray(bsubv, dtype=float)
    pres = np.asarray(pres, dtype=float)
    vp = np.asarray(vp, dtype=float)
    phipf = np.asarray(phipf, dtype=float)
    chipf = np.asarray(chipf, dtype=float)

    buco = np.zeros((ns,), dtype=float)
    bvco = np.zeros((ns,), dtype=float)
    ntheta = int(wint.shape[0])
    nzeta = int(wint.shape[1])
    # Match VMEC's summation order (theta, then zeta) to reduce parity drift.
    for js in range(1, ns):
        acc_u = 0.0
        acc_v = 0.0
        for j in range(ntheta):
            wrow = wint[j]
            bu_row = bsubu[js, j]
            bv_row = bsubv[js, j]
            for k in range(nzeta):
                w = float(wrow[k])
                acc_u += float(bu_row[k]) * w
                acc_v += float(bv_row[k]) * w
        buco[js] = acc_u
        bvco[js] = acc_v

    jcuru = np.zeros((ns,), dtype=float)
    jcurv = np.zeros((ns,), dtype=float)
    vpphi = np.zeros((ns,), dtype=float)
    presgrad = np.zeros((ns,), dtype=float)
    for js in range(1, ns - 1):
        jcurv[js] = float(signgs) * ohs * (buco[js + 1] - buco[js])
        jcuru[js] = -float(signgs) * ohs * (bvco[js + 1] - bvco[js])
        vpphi[js] = 0.5 * (vp[js + 1] + vp[js])
        presgrad[js] = (pres[js + 1] - pres[js]) * ohs

    equif = np.zeros((ns,), dtype=float)
    for js in range(1, ns - 1):
        denom = abs(jcurv[js] * chipf[js]) + abs(jcuru[js] * phipf[js]) + abs(presgrad[js] * vpphi[js])
        if denom != 0.0 and vpphi[js] != 0.0:
            raw = ((-phipf[js] * jcuru[js] + chipf[js] * jcurv[js]) / vpphi[js]) + presgrad[js]
            equif[js] = raw * vpphi[js] / denom

    # Extrapolate endpoints (eqfor.f).
    for arr in (equif, jcuru, jcurv, presgrad, vpphi):
        arr[0] = 2.0 * arr[1] - arr[2]
        arr[-1] = 2.0 * arr[-2] - arr[-3]

    # VMEC stores jcur* in physical units (divide by mu0).
    from .vmec_lforbal import MU0

    jcuru = jcuru / MU0
    jcurv = jcurv / MU0
    return buco, bvco, jcuru, jcurv, equif


def _vmec_realspace_geom_light_from_state(
    *, state: VMECState, modes, trig, lasym: bool | None = None
) -> dict[str, Any]:
    """Compute minimal geometry (R, Z, Zu) needed for wout diagnostics."""
    from ._compat import jnp

    Rcos = jnp.asarray(state.Rcos)
    Rsin = jnp.asarray(state.Rsin)
    Zcos = jnp.asarray(state.Zcos)
    Zsin = jnp.asarray(state.Zsin)
    lthreed = bool(np.any(np.asarray(modes.n)))
    if lasym is None:
        lasym = bool(np.any(np.asarray(Rsin))) or bool(np.any(np.asarray(Zcos)))
    lconm1 = bool(lthreed or lasym)
    if lconm1 and int(np.max(np.asarray(modes.m))) > 0:
        Rcos, Zsin, Rsin, Zcos = vmec_m1_internal_to_physical_signed(
            Rcos=Rcos,
            Zsin=Zsin,
            Rsin=Rsin,
            Zcos=Zcos,
            modes=modes,
            lthreed=lthreed,
            lasym=lasym,
            lconm1=lconm1,
        )
    # Only need the edge surface for aspect ratio; avoid full (ns, ntheta, nzeta)
    # synthesis by slicing to the boundary coefficients.
    coeff_cos_stack = jnp.stack([Rcos, Zcos], axis=0)[:, -1:, :]
    coeff_sin_stack = jnp.stack([Rsin, Zsin], axis=0)[:, -1:, :]
    s_edge = jnp.asarray([1.0], dtype=coeff_cos_stack.dtype)
    rz = vmec_realspace_synthesis(
        coeff_cos=coeff_cos_stack,
        coeff_sin=coeff_sin_stack,
        modes=modes,
        trig=trig,
        coeffs_internal=True,
        apply_scalxc=True,
        s=s_edge,
    )
    rz_t = vmec_realspace_synthesis_dtheta(
        coeff_cos=coeff_cos_stack,
        coeff_sin=coeff_sin_stack,
        modes=modes,
        trig=trig,
        coeffs_internal=True,
        apply_scalxc=True,
        s=s_edge,
    )
    R, Z = rz[0], rz[1]
    Zu = rz_t[1]
    return {"R": R, "Z": Z, "Zu": Zu}


def _apply_bsubv_equif_correction(
    *,
    bsubv: np.ndarray,
    bsubv_e: np.ndarray,
    trig,
) -> np.ndarray:
    """Apply VMEC bcovar iequi=1 correction to bsubv.

    VMEC adjusts the half-mesh bsubv after mesh blending so that the
    surface-average <bsubv> matches the pre-blend current profile fpsi
    (bvco). This routine mirrors the bcovar.f sequence:

      bsubvh(:,js) = 2*bsubv_e(:,js) - bsubvh(:,js+1)
      bsubvh(:,js) += fpsi(js) - SUM(bsubvh(:,js) * pwint(:,js))

    where fpsi(js) is the surface-average of the *pre*-blend bsubv.
    """
    bsubv = np.asarray(bsubv, dtype=float)
    bsubv_e = np.asarray(bsubv_e, dtype=float)
    ns = int(bsubv.shape[0])
    if ns < 3:
        return bsubv

    nzeta = int(bsubv.shape[2])
    pwint = np.asarray(vmec_pwint_from_trig(trig, ns=ns, nzeta=nzeta), dtype=float)
    if pwint.shape != bsubv.shape:
        # Expect (ns, ntheta, nzeta) matching bsubv.
        raise ValueError("pwint shape mismatch in bsubv correction")

    # fpsi = surface-average of pre-blend bsubv (VMEC: bvco)
    fpsi = np.zeros((ns,), dtype=float)
    for js in range(1, ns):
        fpsi[js] = float(np.sum(bsubv[js] * pwint[js]))

    # Start from pre-blend bsubv (half mesh) and update inward using bsubv_e.
    bsubv_h = np.array(bsubv, dtype=float, copy=True)
    for js in range(ns - 2, 0, -1):
        bsubv_h[js] = 2.0 * bsubv_e[js] - bsubv_h[js + 1]

    # Adjust <bsubvh> to fpsi after blending (skip axis).
    for js in range(1, ns):
        curpol = fpsi[js] - float(np.sum(bsubv_h[js] * pwint[js]))
        bsubv_h[js] = bsubv_h[js] + curpol

    return bsubv_h


def _compute_bsubs_half_mesh(**kwargs) -> np.ndarray:
    """Compute bsubs on the half mesh using VMEC's bss.f conventions."""
    return _wout_bsubs_helpers.compute_bsubs_half_mesh(**kwargs)


def _bsubuv_parity_from_state(**kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct parity-separated bsubu/bsubv using VMEC internal even/odd splitting."""
    return _wout_bsubs_helpers.bsubuv_parity_from_state(**kwargs)


def _bsubuv_parity_from_coeffs(**kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split bsubu/bsubv into even/odd m parity using Fourier coefficients."""
    return _wout_bsubs_helpers.bsubuv_parity_from_coeffs(**kwargs)


def _bsubuv_parity_from_realspace_jxbforce(**kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Recover jxbforce parity channels directly from real-space bsubu/bsubv."""
    return _wout_bsubs_helpers.bsubuv_parity_from_realspace_jxbforce(**kwargs)


def _bsubuv_parity_from_bcovar(**kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct parity-separated bsubu/bsubv from bcovar even components."""
    return _wout_bsubs_helpers.bsubuv_parity_from_bcovar(**kwargs)


def _jxbforce_getbsubs_coeffs_lasym_false(
    *,
    frho: np.ndarray,
    bsupu: np.ndarray,
    bsupv: np.ndarray,
    trig,
    nfp: int,
) -> np.ndarray | None:
    """Solve VMEC getbsubs collocation system (lasym=False).

    Returns coefficients ``bsubsmn(m,n)`` with ``m=0..mnyq`` and
    ``n=-nnyq..nnyq`` in the jxbforce convention:
    - ``n>=0``: sin(mu)cos(nv)
    - ``n<0``: cos(mu)sin(|n|v)
    """
    frho = np.asarray(frho, dtype=float)
    bsupu = np.asarray(bsupu, dtype=float)
    bsupv = np.asarray(bsupv, dtype=float)

    nt2 = int(trig.ntheta2)
    nzeta = int(np.asarray(trig.cosnv).shape[0])
    mmax = max(nt2 - 1, 0)
    nmax = max(nzeta // 2, 0)
    if frho.shape != (nt2, nzeta):
        return None
    if bsupu.shape != (nt2, nzeta) or bsupv.shape != (nt2, nzeta):
        return None

    nmax1 = max(0, nmax - 1)
    itotal = nt2 * nzeta - 2 * nmax1
    if itotal <= 0:
        return None

    cosmu = np.asarray(trig.cosmu, dtype=float)[:nt2, : mmax + 1]
    sinmu = np.asarray(trig.sinmu, dtype=float)[:nt2, : mmax + 1]
    cosnv = np.asarray(trig.cosnv, dtype=float)[:, : nmax + 1]
    sinnv = np.asarray(trig.sinnv, dtype=float)[:, : nmax + 1]

    A = np.zeros((itotal, itotal), dtype=float)
    rhs = np.zeros((itotal,), dtype=float)
    row = 0
    z_skip_start = (nzeta // 2) + 1

    for i in range(nt2):
        for k in range(nzeta):
            if (i == 0 or i == nt2 - 1) and (k >= z_skip_start):
                continue
            if row >= itotal:
                return None
            rhs[row] = frho[i, k]
            col = 0
            bu = bsupu[i, k]
            bv = bsupv[i, k]
            for m in range(mmax + 1):
                dm = float(m) * bu
                for n in range(nmax + 1):
                    ccmn = cosmu[i, m] * cosnv[k, n]
                    ssmn = sinmu[i, m] * sinnv[k, n]
                    dn = float(n * nfp) * bv
                    termsc = dm * ccmn - dn * ssmn
                    termcs = -dm * ssmn + dn * ccmn
                    if n == 0 or n == nmax:
                        if m > 0:
                            A[row, col] = termsc
                            col += 1
                        elif n == 0:
                            # Pedestal term for (m,n)=(0,0).
                            A[row, col] = bv
                            col += 1
                        else:
                            A[row, col] = termcs
                            col += 1
                    elif m == 0 or m == mmax:
                        A[row, col] = termcs
                        col += 1
                    else:
                        A[row, col] = termsc
                        col += 1
                        A[row, col] = termcs
                        col += 1
            if col != itotal:
                return None
            row += 1

    if row != itotal:
        return None

    try:
        sol = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        try:
            sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
        except np.linalg.LinAlgError:
            return None

    coeff = np.zeros((mmax + 1, 2 * nmax + 1), dtype=float)
    off = nmax
    idx = 0
    for m in range(mmax + 1):
        for n in range(nmax + 1):
            if n == 0 or n == nmax:
                if m > 0:
                    coeff[m, off + n] = sol[idx]
                    idx += 1
                elif n == 0:
                    coeff[m, off + n] = sol[idx]
                    idx += 1
                else:
                    coeff[m, off - n] = sol[idx]
                    idx += 1
            elif m == 0 or m == mmax:
                coeff[m, off - n] = sol[idx]
                idx += 1
            else:
                coeff[m, off + n] = sol[idx]
                idx += 1
                coeff[m, off - n] = sol[idx]
                idx += 1
    if idx != itotal:
        return None
    return coeff


def _jxbforce_getbsubs_coeffs_lasym_true(
    *,
    frho: np.ndarray,
    bsupu: np.ndarray,
    bsupv: np.ndarray,
    trig,
    nfp: int,
) -> np.ndarray | None:
    """Solve VMEC getbsubs collocation system (lasym=True).

    Port of getbrho.f for the lasym branch. Returns coefficients
    ``bsubsmn(m,n,parity)`` with ``m=0..mmax``, ``n=-nmax..nmax``,
    ``parity=0`` (sin/cos channel) and ``parity=1`` (cos/sin channel).
    """
    frho = np.asarray(frho, dtype=float)
    bsupu = np.asarray(bsupu, dtype=float)
    bsupv = np.asarray(bsupv, dtype=float)

    ntheta3 = int(getattr(trig, "ntheta3", 0))
    ntheta2 = int(getattr(trig, "ntheta2", 0))
    nzeta = int(np.asarray(trig.cosnv).shape[0])
    mmax = max(ntheta2 - 1, 0)
    nmax = max(nzeta // 2, 0)
    if frho.shape != (ntheta3, nzeta):
        return None
    if bsupu.shape != (ntheta3, nzeta) or bsupv.shape != (ntheta3, nzeta):
        return None

    itotal = ntheta3 * nzeta
    if itotal <= 0:
        return None

    cosmu = np.asarray(trig.cosmu, dtype=float)[:ntheta3, : mmax + 1]
    sinmu = np.asarray(trig.sinmu, dtype=float)[:ntheta3, : mmax + 1]
    cosnv = np.asarray(trig.cosnv, dtype=float)[:, : nmax + 1]
    sinnv = np.asarray(trig.sinnv, dtype=float)[:, : nmax + 1]

    A = np.zeros((itotal, itotal), dtype=float)
    rhs = np.zeros((itotal,), dtype=float)
    row = 0

    for i in range(ntheta3):
        for j in range(nzeta):
            rhs[row] = frho[i, j]
            col = 0
            bu = bsupu[i, j]
            bv = bsupv[i, j]
            for m in range(mmax + 1):
                dm = float(m) * bu
                for n in range(nmax + 1):
                    if (m == 0) and (n == 0):
                        # skip (0,0) for lasym in getbsubs
                        continue
                    if col >= itotal:
                        return None
                    ccmn = cosmu[i, m] * cosnv[j, n]
                    ssmn = sinmu[i, m] * sinnv[j, n]
                    dn = float(n * nfp) * bv
                    termsc = dm * ccmn - dn * ssmn
                    termcs = -dm * ssmn + dn * ccmn
                    if n == 0 or n == nmax:
                        if m > 0:
                            A[row, col] = termsc
                            col += 1
                        elif n == 0:
                            A[row, col] = bv  # pedestal term
                            col += 1
                        else:
                            A[row, col] = termcs
                            col += 1
                    elif m == 0 or m == mmax:
                        A[row, col] = termcs
                        col += 1
                    else:
                        A[row, col] = termsc
                        col += 1
                        A[row, col] = termcs
                        col += 1

                    if (m == 0) and (n == 0 or n == nmax):
                        continue
                    if col >= itotal:
                        return None
                    csmn = cosmu[i, m] * sinnv[j, n]
                    scmn = sinmu[i, m] * cosnv[j, n]
                    termcc = -dm * scmn - dn * csmn
                    termss = dm * csmn + dn * scmn
                    if (n == 0 or n == nmax) or (m == 0 or m == mmax):
                        A[row, col] = termcc
                        col += 1
                    else:
                        A[row, col] = termcc
                        col += 1
                        A[row, col] = termss
                        col += 1
            if col != itotal:
                return None
            row += 1

    if row != itotal:
        return None

    try:
        sol = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        try:
            sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
        except np.linalg.LinAlgError:
            return None

    coeff = np.zeros((mmax + 1, 2 * nmax + 1, 2), dtype=float)
    off = nmax
    idx = 0
    for m in range(mmax + 1):
        for n in range(nmax + 1):
            if (m == 0) and (n == 0):
                continue
            if idx >= itotal:
                break
            if n == 0 or n == nmax:
                if m > 0:
                    coeff[m, off + n, 0] = sol[idx]
                    idx += 1
                elif n == 0:
                    coeff[m, off + n, 0] = sol[idx]
                    idx += 1
                else:
                    coeff[m, off - n, 0] = sol[idx]
                    idx += 1
            elif m == 0 or m == mmax:
                coeff[m, off - n, 0] = sol[idx]
                idx += 1
            else:
                coeff[m, off + n, 0] = sol[idx]
                idx += 1
                coeff[m, off - n, 0] = sol[idx]
                idx += 1

            if (m == 0) and (n == 0 or n == nmax):
                continue
            if idx >= itotal:
                break
            if (n == 0 or n == nmax) or (m == 0 or m == mmax):
                coeff[m, off + n, 1] = sol[idx]
                idx += 1
            else:
                coeff[m, off + n, 1] = sol[idx]
                idx += 1
                coeff[m, off - n, 1] = sol[idx]
                idx += 1
    return coeff


def _jxbforce_apply_bsubs_correction_lasym_false(
    *,
    bsubu: np.ndarray,
    bsubv: np.ndarray,
    bsubs: np.ndarray,
    bsubsu: np.ndarray,
    bsubsv: np.ndarray,
    bsupu: np.ndarray,
    bsupv: np.ndarray,
    sqrtg: np.ndarray,
    pres: np.ndarray,
    vp: np.ndarray,
    hs: float,
    signgs: float,
    trig,
    nfp: int,
    sum_w,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror VMEC jxbforce corrected-bsubs pass for lasym=False."""
    bsubu = np.asarray(bsubu, dtype=float)
    bsubv = np.asarray(bsubv, dtype=float)
    bsubs = np.asarray(bsubs, dtype=float)
    bsubsu = np.asarray(bsubsu, dtype=float)
    bsubsv = np.asarray(bsubsv, dtype=float)
    bsupu = np.asarray(bsupu, dtype=float)
    bsupv = np.asarray(bsupv, dtype=float)
    sqrtg = np.asarray(sqrtg, dtype=float)
    pres = np.asarray(pres, dtype=float)
    vp = np.asarray(vp, dtype=float)

    ns, nt2, nzeta = bsubu.shape
    if ns < 3 or hs == 0.0:
        return bsubs, bsubsu, bsubsv

    ohs = 1.0 / hs
    mnyq, nnyq = _jxbforce_nyquist_limits(trig)
    cosmu = np.asarray(trig.cosmu, dtype=float)[:nt2, : mnyq + 1]
    sinmu = np.asarray(trig.sinmu, dtype=float)[:nt2, : mnyq + 1]
    cosmum = np.asarray(trig.cosmum, dtype=float)[:nt2, : mnyq + 1]
    sinmum = np.asarray(trig.sinmum, dtype=float)[:nt2, : mnyq + 1]
    cosnv = np.asarray(trig.cosnv, dtype=float)[:, : nnyq + 1]
    sinnv = np.asarray(trig.sinnv, dtype=float)[:, : nnyq + 1]
    cosnvn = np.asarray(trig.cosnvn, dtype=float)[:, : nnyq + 1]
    sinnvn = np.asarray(trig.sinnvn, dtype=float)[:, : nnyq + 1]

    for js in range(1, ns - 1):
        jxb = 0.5 * (sqrtg[js] + sqrtg[js + 1])
        bsupu1 = 0.5 * (bsupu[js] * sqrtg[js] + bsupu[js + 1] * sqrtg[js + 1])
        bsupv1 = 0.5 * (bsupv[js] * sqrtg[js] + bsupv[js + 1] * sqrtg[js + 1])

        brho = ohs * (bsupu1 * (bsubu[js + 1] - bsubu[js]) + bsupv1 * (bsubv[js + 1] - bsubv[js]))
        brho = brho + (pres[js + 1] - pres[js]) * ohs * jxb

        brho00 = float(sum_w(brho))
        vden = 0.5 * (vp[js] + vp[js + 1])
        if vden != 0.0:
            brho = brho - signgs * jxb * (brho00 / vden)

        coeff = _jxbforce_getbsubs_coeffs_lasym_false(
            frho=brho,
            bsupu=bsupu1,
            bsupv=bsupv1,
            trig=trig,
            nfp=int(nfp),
        )
        if coeff is None:
            continue

        bsubs_s = np.zeros((nt2, nzeta), dtype=float)
        bsubsu_s = np.zeros((nt2, nzeta), dtype=float)
        bsubsv_s = np.zeros((nt2, nzeta), dtype=float)
        off = nnyq

        for m in range(mnyq + 1):
            for n in range(nnyq + 1):
                c1 = coeff[m, off + n]
                c2 = 0.0 if n == 0 else coeff[m, off - n]
                for k in range(nzeta):
                    for j in range(nt2):
                        tsin1 = sinmu[j, m] * cosnv[k, n]
                        tsin2 = cosmu[j, m] * sinnv[k, n]
                        bsubs_s[j, k] += tsin1 * c1 + tsin2 * c2

                        tcosm1 = cosmum[j, m] * cosnv[k, n]
                        tcosm2 = sinmum[j, m] * sinnv[k, n]
                        bsubsu_s[j, k] += tcosm1 * c1 + tcosm2 * c2

                        tcosn1 = sinmu[j, m] * sinnvn[k, n]
                        tcosn2 = cosmu[j, m] * cosnvn[k, n]
                        bsubsv_s[j, k] += tcosn1 * c1 + tcosn2 * c2

        bsubs[js] = bsubs_s
        bsubsu[js] = bsubsu_s
        bsubsv[js] = bsubsv_s

    if ns > 2:
        bsubs[0] = 2.0 * bsubs[1] - bsubs[2]
        bsubs[-1] = 2.0 * bsubs[-1] - bsubs[-2]
    return bsubs, bsubsu, bsubsv


def _jxbforce_apply_bsubs_correction_lasym_true(
    *,
    bsubu: np.ndarray,
    bsubv: np.ndarray,
    bsubs: np.ndarray,
    bsubsu: np.ndarray,
    bsubsv: np.ndarray,
    bsupu: np.ndarray,
    bsupv: np.ndarray,
    sqrtg: np.ndarray,
    pres: np.ndarray,
    vp: np.ndarray,
    hs: float,
    signgs: float,
    trig,
    nfp: int,
    sum_w,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror VMEC jxbforce corrected-bsubs pass for lasym=True."""
    bsubu = np.asarray(bsubu, dtype=float)
    bsubv = np.asarray(bsubv, dtype=float)
    bsubs = np.asarray(bsubs, dtype=float)
    bsubsu = np.asarray(bsubsu, dtype=float)
    bsubsv = np.asarray(bsubsv, dtype=float)
    bsupu = np.asarray(bsupu, dtype=float)
    bsupv = np.asarray(bsupv, dtype=float)
    sqrtg = np.asarray(sqrtg, dtype=float)
    pres = np.asarray(pres, dtype=float)
    vp = np.asarray(vp, dtype=float)

    ns, nt2, nzeta = bsubu.shape
    nt1 = int(getattr(trig, "ntheta1", nt2))
    nt3 = int(getattr(trig, "ntheta3", nt2))
    if nt3 < nt2:
        nt3 = nt2
    if ns < 3 or hs == 0.0:
        return bsubs, bsubsu, bsubsv

    ohs = 1.0 / hs
    mnyq, nnyq = _jxbforce_nyquist_limits(trig)
    cosmu = np.asarray(trig.cosmu, dtype=float)[:nt2, : mnyq + 1]
    sinmu = np.asarray(trig.sinmu, dtype=float)[:nt2, : mnyq + 1]
    cosmum = np.asarray(trig.cosmum, dtype=float)[:nt2, : mnyq + 1]
    sinmum = np.asarray(trig.sinmum, dtype=float)[:nt2, : mnyq + 1]
    cosnv = np.asarray(trig.cosnv, dtype=float)[:, : nnyq + 1]
    sinnv = np.asarray(trig.sinnv, dtype=float)[:, : nnyq + 1]
    cosnvn = np.asarray(trig.cosnvn, dtype=float)[:, : nnyq + 1]
    sinnvn = np.asarray(trig.sinnvn, dtype=float)[:, : nnyq + 1]

    def _expand_sym_to_full(sym: np.ndarray) -> np.ndarray:
        return _vmec_symoutput_expand(sym=sym, asym=None, trig=trig)

    def _extend_parity_to_full(par0: np.ndarray, par1: np.ndarray) -> np.ndarray:
        full = np.zeros((nt3, nzeta), dtype=float)
        full[:nt2, :] = par0 + par1
        if nt3 == nt2:
            return full
        i0 = np.arange(nt2, dtype=int)
        ir0 = np.where(i0 == 0, 0, nt1 - i0)
        kk = (nzeta - np.arange(nzeta, dtype=int)) % nzeta
        mask = ir0 >= nt2
        if np.any(mask):
            ir = ir0[mask]
            ref0 = par0[mask][:, kk]
            ref1 = par1[mask][:, kk]
            full[ir, :] = ref0 - ref1
        return full

    if bsubu.shape[1] == nt3:
        bsubu_full = bsubu
        bsubv_full = bsubv
    else:
        bsubu_full = _expand_sym_to_full(bsubu)
        bsubv_full = _expand_sym_to_full(bsubv)

    if bsupu.shape[1] == nt3:
        bsupu_full = bsupu
        bsupv_full = bsupv
    else:
        bsupu_full = _expand_sym_to_full(bsupu[:, :nt2, :])
        bsupv_full = _expand_sym_to_full(bsupv[:, :nt2, :])

    if sqrtg.shape[1] == nt3:
        sqrtg_full = sqrtg
    else:
        sqrtg_full = _expand_sym_to_full(sqrtg[:, :nt2, :])

    if bsubs.shape[1] == nt3:
        bsubs_out = bsubs.copy()
    else:
        bsubs_out = np.zeros((ns, nt3, nzeta), dtype=float)
    bsubsu_out = np.zeros((ns, nt3, nzeta), dtype=float)
    bsubsv_out = np.zeros((ns, nt3, nzeta), dtype=float)

    for js in range(1, ns - 1):
        jxb = 0.5 * (sqrtg_full[js] + sqrtg_full[js + 1])
        bsupu1 = 0.5 * (bsupu_full[js] * sqrtg_full[js] + bsupu_full[js + 1] * sqrtg_full[js + 1])
        bsupv1 = 0.5 * (bsupv_full[js] * sqrtg_full[js] + bsupv_full[js + 1] * sqrtg_full[js + 1])

        brho_full = ohs * (
            bsupu1 * (bsubu_full[js + 1] - bsubu_full[js]) + bsupv1 * (bsubv_full[js + 1] - bsubv_full[js])
        )
        brho_full = brho_full + (pres[js + 1] - pres[js]) * ohs * jxb

        brho00 = float(sum_w(brho_full))
        vden = 0.5 * (vp[js] + vp[js + 1])
        if vden != 0.0:
            brho_full = brho_full - signgs * jxb * (brho00 / vden)

        coeff = _jxbforce_getbsubs_coeffs_lasym_true(
            frho=brho_full,
            bsupu=bsupu1,
            bsupv=bsupv1,
            trig=trig,
            nfp=int(nfp),
        )
        if coeff is None:
            continue

        bsubs_s = np.zeros((nt2, nzeta), dtype=float)
        bsubsu_s = np.zeros((nt2, nzeta), dtype=float)
        bsubsv_s = np.zeros((nt2, nzeta), dtype=float)
        bsubs_a = np.zeros((nt2, nzeta), dtype=float)
        bsubsu_a = np.zeros((nt2, nzeta), dtype=float)
        bsubsv_a = np.zeros((nt2, nzeta), dtype=float)
        off = nnyq

        for m in range(mnyq + 1):
            for n in range(nnyq + 1):
                c1 = coeff[m, off + n, 0]
                c2 = 0.0 if n == 0 else coeff[m, off - n, 0]
                c3 = coeff[m, off + n, 1]
                c4 = 0.0 if n == 0 else coeff[m, off - n, 1]
                for k in range(nzeta):
                    for j in range(nt2):
                        tsin1 = sinmu[j, m] * cosnv[k, n]
                        tsin2 = cosmu[j, m] * sinnv[k, n]
                        bsubs_s[j, k] += tsin1 * c1 + tsin2 * c2
                        tcosm1 = cosmum[j, m] * cosnv[k, n]
                        tcosm2 = sinmum[j, m] * sinnv[k, n]
                        bsubsu_s[j, k] += tcosm1 * c1 + tcosm2 * c2
                        tcosn1 = sinmu[j, m] * sinnvn[k, n]
                        tcosn2 = cosmu[j, m] * cosnvn[k, n]
                        bsubsv_s[j, k] += tcosn1 * c1 + tcosn2 * c2

                        tcos1 = cosmu[j, m] * cosnv[k, n]
                        tcos2 = sinmu[j, m] * sinnv[k, n]
                        bsubs_a[j, k] += tcos1 * c3 + tcos2 * c4
                        tsinm1 = sinmum[j, m] * cosnv[k, n]
                        tsinm2 = cosmum[j, m] * sinnv[k, n]
                        bsubsu_a[j, k] += tsinm1 * c3 + tsinm2 * c4
                        tsinn1 = cosmu[j, m] * sinnvn[k, n]
                        tsinn2 = sinmu[j, m] * cosnvn[k, n]
                        bsubsv_a[j, k] += tsinn1 * c3 + tsinn2 * c4

        bsubs_full = _extend_parity_to_full(bsubs_a, bsubs_s)
        bsubs_out[js] = bsubs_full
        bsubsu_out[js] = _extend_parity_to_full(bsubsu_s, bsubsu_a)
        bsubsv_out[js] = _extend_parity_to_full(bsubsv_s, bsubsv_a)

    if ns > 2:
        bsubs_out[0] = 2.0 * bsubs_out[1] - bsubs_out[2]
        bsubs_out[-1] = 2.0 * bsubs_out[-1] - bsubs_out[-2]
    return bsubs_out, bsubsu_out, bsubsv_out



def _compute_mercier(**kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compatibility wrapper for the WOUT Mercier/JXBFORCE reducer."""
    return _wout_mercier.compute_mercier(
        **kwargs,
        compute_bsubs_half_mesh=_compute_bsubs_half_mesh,
        symoutput_expand=_vmec_symoutput_expand,
        filter_bsubuv_jxbforce_lasym_loop=_filter_bsubuv_jxbforce_lasym_loop,
        pshalf_from_s=_pshalf_from_s,
        jxbforce_filter_with_bsubs_derivs_loop=_jxbforce_filter_with_bsubs_derivs_loop,
        jxbforce_bsubsu_bsubsv_loop=_jxbforce_bsubsu_bsubsv_loop,
        symoutput_split=_vmec_symoutput_split,
        vmec_wint_from_trig=_vmec_wint_from_trig,
        jxbforce_apply_bsubs_correction_lasym_true=_jxbforce_apply_bsubs_correction_lasym_true,
        jxbforce_apply_bsubs_correction_lasym_false=_jxbforce_apply_bsubs_correction_lasym_false,
    )


def read_wout(path: str | Path) -> WoutData:
    """Read a subset of `wout_*.nc` needed for regression comparisons."""
    path = Path(path)
    try:
        import netCDF4  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("netCDF4 is required to read wout files (pip install vmec-jax)") from e

    with netCDF4.Dataset(path) as ds:
        ns, mpol, ntor, nfp, lasym, signgs = _read_wout_scalar_metadata(ds.variables, path=path)

        xm = read_mode_table(ds.variables, "xm", path=path)
        xn = read_mode_table(ds.variables, "xn", path=path)
        xm_nyq = read_mode_table(ds.variables, "xm_nyq", path=path)
        xn_nyq = read_mode_table(ds.variables, "xn_nyq", path=path)
        mpol_nyq_default = int(np.max(xm_nyq)) if xm_nyq.size else 0
        ntor_nyq_default = int(np.max(np.abs(xn_nyq // nfp))) if xn_nyq.size else 0
        mnmax = read_optional_int_scalar(ds.variables, "mnmax", xm.size)
        mnmax_nyq = read_optional_int_scalar(ds.variables, "mnmax_nyq", xm_nyq.size)
        mpol_nyq = read_optional_int_scalar(ds.variables, "mpol_nyq", mpol_nyq_default)
        ntor_nyq = read_optional_int_scalar(ds.variables, "ntor_nyq", ntor_nyq_default)

        rmnc = np.asarray(ds.variables["rmnc"][:])
        rmns = np.asarray(ds.variables.get("rmns", np.zeros_like(rmnc))[:])
        zmns = np.asarray(ds.variables["zmns"][:])
        zmnc = np.asarray(ds.variables.get("zmnc", np.zeros_like(zmns))[:])
        lmns = np.asarray(ds.variables["lmns"][:])
        lmnc = np.asarray(ds.variables.get("lmnc", np.zeros_like(lmns))[:])

        phipf = np.asarray(ds.variables["phipf"][:])
        chipf = np.asarray(ds.variables["chipf"][:])
        phips = np.asarray(ds.variables["phips"][:])
        iotaf = np.asarray(ds.variables.get("iotaf", np.zeros_like(phips))[:])
        iotas = np.asarray(ds.variables.get("iotas", np.zeros_like(phips))[:])

        nyquist_fields = read_nyquist_fourier_fields(ds.variables)

        wb = float(_nc_scalar(ds.variables["wb"][:], 0.0))
        volume_p = float(_nc_scalar(ds.variables["volume_p"][:], 0.0))
        gamma = float(_nc_scalar(ds.variables.get("gamma", 0.0)[:], 0.0)) if "gamma" in ds.variables else 0.0
        wp = float(_nc_scalar(ds.variables.get("wp", 0.0)[:], 0.0)) if "wp" in ds.variables else 0.0
        vp = np.asarray(ds.variables.get("vp", np.zeros((ns,), dtype=float))[:])

        # `wout` stores pres/presf divided by mu0. Convert back to VMEC internal
        # units (mu0*Pa) so it matches the energy functional.
        pres_pa = np.asarray(ds.variables.get("pres", np.zeros((ns,), dtype=float))[:])
        presf_pa = np.asarray(ds.variables.get("presf", np.zeros((ns,), dtype=float))[:])
        pres = MU0 * pres_pa
        presf = MU0 * presf_pa

        # Force residual scalars (present in most VMEC wout files).
        fsqr = float(_nc_scalar(ds.variables.get("fsqr", 0.0)[:], 0.0)) if "fsqr" in ds.variables else 0.0
        fsqz = float(_nc_scalar(ds.variables.get("fsqz", 0.0)[:], 0.0)) if "fsqz" in ds.variables else 0.0
        fsql = float(_nc_scalar(ds.variables.get("fsql", 0.0)[:], 0.0)) if "fsql" in ds.variables else 0.0
        fsqt = np.asarray(ds.variables.get("fsqt", np.zeros((0,), dtype=float))[:])
        equif = np.asarray(ds.variables.get("equif", np.zeros((ns,), dtype=float))[:])

        # Additional fields used by vmecPlot2 and diagnostics.
        phi = _wout_phi_profile_from_variables(ds.variables, ns=ns, phipf=phipf)

        buco = np.asarray(ds.variables.get("buco", np.zeros((ns,), dtype=float))[:])
        bvco = np.asarray(ds.variables.get("bvco", np.zeros((ns,), dtype=float))[:])
        jcuru = np.asarray(ds.variables.get("jcuru", np.zeros((ns,), dtype=float))[:])
        jcurv = np.asarray(ds.variables.get("jcurv", np.zeros((ns,), dtype=float))[:])

        raxis_cc = np.asarray(ds.variables.get("raxis_cc", np.zeros((ntor + 1,), dtype=float))[:])
        zaxis_cs = np.asarray(ds.variables.get("zaxis_cs", np.zeros((ntor + 1,), dtype=float))[:])
        raxis_cs = np.asarray(ds.variables.get("raxis_cs", np.zeros_like(raxis_cc))[:])
        zaxis_cc = np.asarray(ds.variables.get("zaxis_cc", np.zeros_like(zaxis_cs))[:])

        Aminor_p = float(_nc_scalar(ds.variables.get("Aminor_p", 0.0)[:], 0.0)) if "Aminor_p" in ds.variables else 0.0
        Rmajor_p = float(_nc_scalar(ds.variables.get("Rmajor_p", 0.0)[:], 0.0)) if "Rmajor_p" in ds.variables else 0.0
        aspect = float(_nc_scalar(ds.variables.get("aspect", 0.0)[:], 0.0)) if "aspect" in ds.variables else 0.0
        betatotal = (
            float(_nc_scalar(ds.variables.get("betatotal", 0.0)[:], 0.0)) if "betatotal" in ds.variables else 0.0
        )
        betapol = float(_nc_scalar(ds.variables.get("betapol", 0.0)[:], 0.0)) if "betapol" in ds.variables else 0.0
        betator = float(_nc_scalar(ds.variables.get("betator", 0.0)[:], 0.0)) if "betator" in ds.variables else 0.0
        betaxis = float(_nc_scalar(ds.variables.get("betaxis", 0.0)[:], 0.0)) if "betaxis" in ds.variables else 0.0
        ctor = float(_nc_scalar(ds.variables.get("ctor", 0.0)[:], 0.0)) if "ctor" in ds.variables else 0.0

        DMerc = np.asarray(ds.variables.get("DMerc", np.zeros((ns,), dtype=float))[:])
        Dshear = np.asarray(ds.variables.get("DShear", np.zeros((ns,), dtype=float))[:])
        Dwell = np.asarray(ds.variables.get("DWell", np.zeros((ns,), dtype=float))[:])
        Dcurr = np.asarray(ds.variables.get("DCurr", np.zeros((ns,), dtype=float))[:])
        Dgeod = np.asarray(ds.variables.get("DGeod", np.zeros((ns,), dtype=float))[:])
        jdotb = np.asarray(ds.variables.get("jdotb", np.zeros((ns,), dtype=float))[:])
        bdotb = np.asarray(ds.variables.get("bdotb", np.zeros((ns,), dtype=float))[:])
        bdotgradv = np.asarray(ds.variables.get("bdotgradv", np.zeros((ns,), dtype=float))[:])
        glasser_profiles = _glasser_profiles_from_wout_variables(
            ds.variables,
            DMerc=DMerc,
            Dshear=Dshear,
            Dcurr=Dcurr,
        )
        D_R = glasser_profiles.D_R
        H_glasser = glasser_profiles.H
        glasser_correction = glasser_profiles.correction
        glasser_shear_valid = glasser_profiles.shear_valid

        ac = np.asarray(ds.variables.get("ac", np.zeros((0,), dtype=float))[:])
        ac_aux_s = np.asarray(ds.variables.get("ac_aux_s", -np.ones((101,), dtype=float))[:])
        ac_aux_f = np.asarray(ds.variables.get("ac_aux_f", np.zeros((101,), dtype=float))[:])

        pcurr_type = read_type_field(ds.variables, "pcurr_type")
        piota_type = read_type_field(ds.variables, "piota_type")
        ier_flag = read_optional_int_scalar(ds.variables, "ier_flag", 0)
        vmec_jax_converged_var = ds.variables.get("vmec_jax_converged__logical__")
        if vmec_jax_converged_var is None:
            vmec_jax_converged = ier_flag == 0
        else:
            vmec_jax_converged = _bool_from_nc(vmec_jax_converged_var[:])
        vmec_jax_status = read_type_field(ds.variables, "vmec_jax_status")
        if not vmec_jax_status:
            vmec_jax_status = "converged" if bool(vmec_jax_converged) else "nonconverged"

    return WoutData(
        path=path,
        ns=ns,
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        lasym=lasym,
        signgs=signgs,
        mnmax=mnmax,
        mpol_nyq=mpol_nyq,
        ntor_nyq=ntor_nyq,
        mnmax_nyq=mnmax_nyq,
        xm=xm,
        xn=xn,
        xm_nyq=xm_nyq,
        xn_nyq=xn_nyq,
        rmnc=rmnc,
        rmns=rmns,
        zmnc=zmnc,
        zmns=zmns,
        lmnc=lmnc,
        lmns=lmns,
        phipf=phipf,
        chipf=chipf,
        phips=phips,
        iotaf=iotaf,
        iotas=iotas,
        **nyquist_fields,
        wb=wb,
        volume_p=volume_p,
        gamma=gamma,
        wp=wp,
        vp=vp,
        pres=pres,
        presf=presf,
        fsqr=fsqr,
        fsqz=fsqz,
        fsql=fsql,
        fsqt=fsqt,
        equif=equif,
        phi=phi,
        buco=buco,
        bvco=bvco,
        jcuru=jcuru,
        jcurv=jcurv,
        raxis_cc=raxis_cc,
        zaxis_cs=zaxis_cs,
        raxis_cs=raxis_cs,
        zaxis_cc=zaxis_cc,
        Aminor_p=Aminor_p,
        Rmajor_p=Rmajor_p,
        aspect=aspect,
        betatotal=betatotal,
        betapol=betapol,
        betator=betator,
        betaxis=betaxis,
        ctor=ctor,
        DMerc=DMerc,
        Dshear=Dshear,
        Dwell=Dwell,
        Dcurr=Dcurr,
        Dgeod=Dgeod,
        D_R=D_R,
        H=H_glasser,
        glasser_correction=glasser_correction,
        glasser_shear_valid=glasser_shear_valid,
        jdotb=jdotb,
        bdotb=bdotb,
        bdotgradv=bdotgradv,
        ac=ac,
        ac_aux_s=ac_aux_s,
        ac_aux_f=ac_aux_f,
        pcurr_type=pcurr_type,
        piota_type=piota_type,
        ier_flag=ier_flag,
        vmec_jax_converged=bool(vmec_jax_converged),
        vmec_jax_status=vmec_jax_status,
    )


def write_wout(path: str | Path, wout: WoutData, *, overwrite: bool = False) -> None:
    """Write a minimal VMEC-style ``wout_*.nc`` file.

    This is intended for:
    - round-tripping reference ``wout`` files (read -> write -> read),
    - emitting VMEC-compatible output containers from vmec_jax as parity work progresses.

    Notes
    -----
    - Only the subset of variables represented in :class:`WoutData` is written.
    - ``pres`` and ``presf`` are written in **Pa** (VMEC convention: netCDF stores
      pressure divided by ``mu0``). Internally :class:`WoutData` stores pressure in
      VMEC internal units (``mu0*Pa``).
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists (pass overwrite=True to overwrite)")

    try:
        import netCDF4  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("netCDF4 is required to write wout files (pip install vmec-jax)") from e

    # Dimensions.
    ns = int(wout.ns)
    mnmax = int(np.asarray(wout.xm).size)
    mnmax_nyq = int(np.asarray(wout.xm_nyq).size)
    nstore = int(np.asarray(wout.fsqt).size)
    n_tor = int(wout.ntor) + 1
    ac = np.asarray(getattr(wout, "ac", np.zeros((0,), dtype=float)))
    if ac.size == 0:
        ac = np.zeros((21,), dtype=float)
    ac_aux_s = np.asarray(getattr(wout, "ac_aux_s", -np.ones((101,), dtype=float)))
    ac_aux_f = np.asarray(getattr(wout, "ac_aux_f", np.zeros((101,), dtype=float)))
    if ac_aux_s.size == 0:
        ac_aux_s = -np.ones((1,), dtype=float)
    if ac_aux_f.size == 0:
        ac_aux_f = np.zeros((1,), dtype=float)
    ndfmax = int(ac_aux_s.size)
    preset = int(ac.size)

    # Convert pressures back to VMEC netcdf convention (Pa).
    pres_pa = np.asarray(wout.pres) / MU0
    presf_pa = np.asarray(wout.presf) / MU0

    # Use VMEC-like dimension names for better interoperability with external tools.
    with netCDF4.Dataset(path, mode="w", format="NETCDF3_CLASSIC") as ds:
        # Avoid pre-filling data to speed up large writes.
        try:
            ds.set_fill_off()
        except Exception as exc:
            if os.getenv("VMEC_JAX_MERCIER_RAISE", "") not in ("", "0", "false", "no"):
                raise
            if os.getenv("VMEC_JAX_MERCIER_LOG", "") not in ("", "0", "false", "no"):
                print(f"[vmec_jax] Mercier/jdotb computation failed: {exc}", flush=True)
        ds.createDimension("radius", ns)
        ds.createDimension("mn_mode", mnmax)
        ds.createDimension("mn_mode_nyq", mnmax_nyq)
        ds.createDimension("nstore_seq", nstore)
        ds.createDimension("n_tor", n_tor)
        ds.createDimension("ndfmax", ndfmax)
        ds.createDimension("preset", preset)
        ds.createDimension("dim_00020", 20)

        # Scalars.
        write_int_variable(ds, "ns", (), np.asarray(ns))
        write_int_variable(ds, "mpol", (), np.asarray(int(wout.mpol)))
        write_int_variable(ds, "ntor", (), np.asarray(int(wout.ntor)))
        write_int_variable(ds, "nfp", (), np.asarray(int(wout.nfp)))
        write_int_variable(ds, "signgs", (), np.asarray(int(wout.signgs)))
        write_int_variable(ds, "lasym__logical__", (), np.asarray(int(bool(wout.lasym))))
        wout_converged = bool(getattr(wout, "vmec_jax_converged", True))
        ier_flag = int(getattr(wout, "ier_flag", 0 if wout_converged else 1))
        write_int_variable(ds, "ier_flag", (), np.asarray(ier_flag))
        write_int_variable(ds, "vmec_jax_converged__logical__", (), np.asarray(int(wout_converged)))
        write_int_variable(ds, "mnmax", (), np.asarray(int(getattr(wout, "mnmax", mnmax))))
        write_int_variable(
            ds,
            "mpol_nyq",
            (),
            np.asarray(int(getattr(wout, "mpol_nyq", np.max(np.asarray(wout.xm_nyq)) if mnmax_nyq > 0 else 0))),
        )
        write_int_variable(
            ds,
            "ntor_nyq",
            (),
            np.asarray(
                int(
                    getattr(
                        wout,
                        "ntor_nyq",
                        np.max(np.abs(np.asarray(wout.xn_nyq) // int(wout.nfp))) if mnmax_nyq > 0 else 0,
                    )
                )
            ),
        )
        write_int_variable(ds, "mnmax_nyq", (), np.asarray(int(getattr(wout, "mnmax_nyq", mnmax_nyq))))

        write_float_variable(ds, "wb", (), np.asarray(float(wout.wb)))
        write_float_variable(ds, "volume_p", (), np.asarray(float(wout.volume_p)))
        write_float_variable(ds, "gamma", (), np.asarray(float(wout.gamma)))
        write_float_variable(ds, "wp", (), np.asarray(float(wout.wp)))
        write_float_variable(ds, "fsqr", (), np.asarray(float(wout.fsqr)))
        write_float_variable(ds, "fsqz", (), np.asarray(float(wout.fsqz)))
        write_float_variable(ds, "fsql", (), np.asarray(float(wout.fsql)))

        # Mode tables.
        # Keep the scalar mode-count metadata as integers, but store the mode
        # tables themselves as floats to match the legacy libstell/SFINCS wout
        # convention while remaining readable by integer-oriented consumers.
        write_float_variable(ds, "xm", ("mn_mode",), np.asarray(wout.xm))
        write_float_variable(ds, "xn", ("mn_mode",), np.asarray(wout.xn))
        write_float_variable(ds, "xm_nyq", ("mn_mode_nyq",), np.asarray(wout.xm_nyq))
        write_float_variable(ds, "xn_nyq", ("mn_mode_nyq",), np.asarray(wout.xn_nyq))

        # Geometry coefficients (full mesh).
        write_float_variable(ds, "rmnc", ("radius", "mn_mode"), np.asarray(wout.rmnc))
        write_float_variable(ds, "rmns", ("radius", "mn_mode"), np.asarray(wout.rmns))
        write_float_variable(ds, "zmnc", ("radius", "mn_mode"), np.asarray(wout.zmnc))
        write_float_variable(ds, "zmns", ("radius", "mn_mode"), np.asarray(wout.zmns))
        write_float_variable(ds, "lmnc", ("radius", "mn_mode"), np.asarray(wout.lmnc))
        write_float_variable(ds, "lmns", ("radius", "mn_mode"), np.asarray(wout.lmns))

        # Flux functions / profiles.
        write_float_variable(ds, "phipf", ("radius",), np.asarray(wout.phipf))
        write_float_variable(ds, "chipf", ("radius",), np.asarray(wout.chipf))
        write_float_variable(ds, "phips", ("radius",), np.asarray(wout.phips))
        write_float_variable(ds, "iotaf", ("radius",), np.asarray(wout.iotaf))
        write_float_variable(ds, "iotas", ("radius",), np.asarray(wout.iotas))
        write_float_variable(ds, "phi", ("radius",), np.asarray(getattr(wout, "phi", np.zeros((ns,), dtype=float))))

        # Nyquist Fourier fields.
        write_nyquist_fourier_fields(ds, wout)

        # 1D radial fields.
        write_float_variable(ds, "vp", ("radius",), np.asarray(wout.vp))
        write_float_variable(ds, "pres", ("radius",), np.asarray(pres_pa))
        write_float_variable(ds, "presf", ("radius",), np.asarray(presf_pa))
        write_float_variable(ds, "equif", ("radius",), np.asarray(getattr(wout, "equif", np.zeros((ns,), dtype=float))))
        write_float_variable(ds, "buco", ("radius",), np.asarray(getattr(wout, "buco", np.zeros((ns,), dtype=float))))
        write_float_variable(ds, "bvco", ("radius",), np.asarray(getattr(wout, "bvco", np.zeros((ns,), dtype=float))))
        write_float_variable(ds, "jcuru", ("radius",), np.asarray(getattr(wout, "jcuru", np.zeros((ns,), dtype=float))))
        write_float_variable(ds, "jcurv", ("radius",), np.asarray(getattr(wout, "jcurv", np.zeros((ns,), dtype=float))))
        write_float_variable(ds, "jdotb", ("radius",), np.asarray(getattr(wout, "jdotb", np.zeros((ns,), dtype=float))))
        write_float_variable(ds, "bdotb", ("radius",), np.asarray(getattr(wout, "bdotb", np.zeros((ns,), dtype=float))))
        write_float_variable(
            ds, "bdotgradv", ("radius",), np.asarray(getattr(wout, "bdotgradv", np.zeros((ns,), dtype=float)))
        )
        write_float_variable(ds, "DMerc", ("radius",), np.asarray(getattr(wout, "DMerc", np.zeros((ns,), dtype=float))))
        write_float_variable(
            ds, "DShear", ("radius",), np.asarray(getattr(wout, "Dshear", np.zeros((ns,), dtype=float)))
        )
        write_float_variable(ds, "DWell", ("radius",), np.asarray(getattr(wout, "Dwell", np.zeros((ns,), dtype=float))))
        write_float_variable(ds, "DCurr", ("radius",), np.asarray(getattr(wout, "Dcurr", np.zeros((ns,), dtype=float))))
        write_float_variable(ds, "DGeod", ("radius",), np.asarray(getattr(wout, "Dgeod", np.zeros((ns,), dtype=float))))
        glasser_profiles = _glasser_profiles_from_wout_data(wout, ns)
        write_float_variable(ds, "D_R", ("radius",), glasser_profiles.D_R)
        write_float_variable(ds, "HGlasser", ("radius",), glasser_profiles.H)
        write_float_variable(ds, "GlasserCorrection", ("radius",), glasser_profiles.correction)
        write_float_variable(ds, "GlasserShearValid", ("radius",), np.asarray(glasser_profiles.shear_valid, dtype=float))

        # Iteration trace (optional).
        write_float_variable(ds, "fsqt", ("nstore_seq",), np.asarray(wout.fsqt))

        # Axis coefficients and geometric scalars.
        write_float_variable(
            ds, "raxis_cc", ("n_tor",), np.asarray(getattr(wout, "raxis_cc", np.zeros((n_tor,), dtype=float)))
        )
        write_float_variable(
            ds, "zaxis_cs", ("n_tor",), np.asarray(getattr(wout, "zaxis_cs", np.zeros((n_tor,), dtype=float)))
        )
        write_float_variable(
            ds, "raxis_cs", ("n_tor",), np.asarray(getattr(wout, "raxis_cs", np.zeros((n_tor,), dtype=float)))
        )
        write_float_variable(
            ds, "zaxis_cc", ("n_tor",), np.asarray(getattr(wout, "zaxis_cc", np.zeros((n_tor,), dtype=float)))
        )

        write_float_variable(ds, "Aminor_p", (), np.asarray(float(getattr(wout, "Aminor_p", 0.0))))
        write_float_variable(ds, "Rmajor_p", (), np.asarray(float(getattr(wout, "Rmajor_p", 0.0))))
        write_float_variable(ds, "aspect", (), np.asarray(float(getattr(wout, "aspect", 0.0))))
        write_float_variable(ds, "betatotal", (), np.asarray(float(getattr(wout, "betatotal", 0.0))))
        write_float_variable(ds, "betapol", (), np.asarray(float(getattr(wout, "betapol", 0.0))))
        write_float_variable(ds, "betator", (), np.asarray(float(getattr(wout, "betator", 0.0))))
        write_float_variable(ds, "betaxis", (), np.asarray(float(getattr(wout, "betaxis", 0.0))))
        write_float_variable(ds, "ctor", (), np.asarray(float(getattr(wout, "ctor", 0.0))))

        write_float_variable(ds, "ac_aux_s", ("ndfmax",), np.asarray(ac_aux_s))
        write_float_variable(ds, "ac_aux_f", ("ndfmax",), np.asarray(ac_aux_f))
        write_float_variable(ds, "ac", ("preset",), np.asarray(ac))

        write_fixed_width_string_variable(ds, "pcurr_type", getattr(wout, "pcurr_type", ""))
        write_fixed_width_string_variable(ds, "piota_type", getattr(wout, "piota_type", ""))
        write_fixed_width_string_variable(
            ds,
            "vmec_jax_status",
            getattr(wout, "vmec_jax_status", "converged" if wout_converged else "nonconverged"),
        )


def wout_minimal_from_fixed_boundary(
    *,
    path: str | Path,
    state: VMECState,
    static,
    indata,
    signgs: int,
    fsqr: float,
    fsqz: float,
    fsql: float,
    fsqt: np.ndarray | None = None,
    converged: bool | None = None,
    flux_override=None,
    profiles_override: dict | None = None,
    force_payload_override=None,
) -> WoutData:
    """Build a minimal :class:`WoutData` from an input-only fixed-boundary run.

    This helper is intended for producing VMEC-compatible output files from
    vmec_jax *without* reading any existing `wout_*.nc` as an input.

    Scope:
    - Writes the main Fourier coefficients (R/Z/lambda) using `vmec_mode_table`.
    - Writes flux functions and profiles derived from `indata` (same path used by the solver).
    - Sets Nyquist-derived fields (gmnc/bsup*/bsub*/bmnc) to zeros for now.
      These can be filled in later once the full VMEC nyquist output path is
      fully ported end-to-end.
    """
    from .energy import flux_profiles_from_indata
    from .profiles import eval_profiles
    from .integrals import cumrect_s_halfmesh
    from .vmec_tomnsp import vmec_trig_tables
    from .vmec_bcovar import vmec_bcovar_half_mesh_from_wout
    from .vmec_residue import vmec_force_norms_from_bcovar_dynamic

    cfg = static.cfg
    ns = int(cfg.ns)
    mpol = int(cfg.mpol)
    ntor = int(cfg.ntor)
    nfp = int(cfg.nfp)
    lasym = bool(cfg.lasym)

    wout_timing_env = os.getenv("VMEC_JAX_WOUT_TIMING", "").strip().lower()
    wout_timing_enabled = wout_timing_env not in ("", "0", "false", "no")
    wout_light_env = os.getenv("VMEC_JAX_WOUT_LIGHT", "").strip().lower()
    wout_light = wout_light_env not in ("", "0", "false", "no")
    wout_fast_bcovar_env = os.getenv("VMEC_JAX_WOUT_FAST_BCOVAR", "").strip().lower()
    wout_fast_bcovar = wout_fast_bcovar_env not in ("0", "false", "no", "off")
    if wout_light:
        # Light output favors speed; also use the fast bcovar path.
        wout_fast_bcovar = True
    wout_timing: dict[str, float] = {}
    if wout_timing_enabled:
        import time as _time

        t_wout_total_start = _time.perf_counter()

    if converged is None:
        converged = True

    # VMEC2000 uses namelist LBSUBS to enable the "corrected" bsubs path
    # in jxbforce. Default is False in libstell vmec_input.
    lbsubs = bool(getattr(indata, "get_bool", lambda *_args, **_kwargs: False)("LBSUBS", False))
    # Allow explicit env override for parity/debugging.
    _lbsubs_env = os.getenv("VMEC_JAX_ENABLE_BSUBS_CORR", "").strip().lower()
    if _lbsubs_env not in ("", "0", "false", "no"):
        lbsubs = True

    main_modes = vmec_mode_table(mpol, ntor)
    if int(main_modes.K) != int(state.layout.K):
        raise ValueError("state mode count does not match vmec_mode_table(mpol,ntor)")

    nyq_modes = nyquist_mode_table_from_grid(
        mpol=mpol,
        ntor=ntor,
        ntheta=int(cfg.ntheta),
        nzeta=int(cfg.nzeta),
    )

    mmax_nyq = int(np.max(nyq_modes.m)) if int(nyq_modes.K) > 0 else 0
    nmax_nyq = int(np.max(np.abs(nyq_modes.n))) if int(nyq_modes.K) > 0 else 0
    mmax_base = max(int(mpol) - 1, mmax_nyq)
    nmax_base = max(int(ntor), nmax_nyq)
    if wout_timing_enabled:
        t0 = _time.perf_counter()
    trig = vmec_trig_tables(
        ntheta=int(cfg.ntheta),
        nzeta=int(cfg.nzeta),
        nfp=int(nfp),
        mmax=int(mmax_base),
        nmax=int(nmax_base),
        lasym=bool(lasym),
        dtype=np.asarray(state.Rcos).dtype,
    )
    if wout_timing_enabled:
        wout_timing["trig_tables_s"] = _time.perf_counter() - t0

    if wout_timing_enabled:
        t0 = _time.perf_counter()
    from .vmec_numpy_forces import _numpy_module_patch

    with _numpy_module_patch():
        if wout_light:
            geom = _vmec_realspace_geom_light_from_state(state=state, modes=static.modes, trig=trig)
        else:
            geom = vmec_realspace_geom_from_state(state=state, modes=static.modes, trig=trig)
    if has_jax():
        try:
            geom = jax.device_get(geom)
        except Exception:
            pass
    geom = {k: (None if v is None else np.asarray(v)) for k, v in geom.items()}
    if wout_timing_enabled:
        wout_timing["geom_synthesis_s"] = _time.perf_counter() - t0

    # Flux and profiles on VMEC half mesh.
    s = np.asarray(static.s)
    flux = flux_override if flux_override is not None else flux_profiles_from_indata(indata, s, signgs=int(signgs))
    chipf_wout = np.asarray(flux.chipf)
    phips = np.asarray(flux.phips)
    if phips.size:
        phips = phips.copy()
        phips[0] = 0.0

    if ns < 2:
        s_half = s
    else:
        s_half = np.concatenate([s[:1], 0.5 * (s[1:] + s[:-1])], axis=0)
    prof = dict(profiles_override) if profiles_override is not None else eval_profiles(indata, s_half)
    pres = np.asarray(prof.get("pressure", np.zeros((ns,), dtype=float)))
    if pres.size:
        pres = pres.copy()
        pres[0] = 0.0
    # VMEC mass profile: mass = pmass * (|vnorm|*r00)^gamma
    from .boundary import boundary_from_indata

    boundary = boundary_from_indata(indata, main_modes)
    idx00 = np.where((np.asarray(main_modes.m) == 0) & (np.asarray(main_modes.n) == 0))[0]
    r00 = float(boundary.R_cos[int(idx00[0])]) if idx00.size else float(np.asarray(boundary.R_cos)[0])
    gamma = float(indata.get_float("GAMMA", 0.0))
    lrfp = bool(indata.get_bool("LRFP", False))
    vnorm = phips
    if lrfp:
        # For RFP, use poloidal flux derivative instead of phips.
        chipf = np.asarray(flux.chipf)
        if chipf.size:
            chips = np.concatenate([chipf[:1], 0.5 * (chipf[1:] + chipf[:-1])], axis=0)
            vnorm = chips
    mass = pres * (np.abs(vnorm) * r00) ** gamma
    if mass.size:
        mass = mass.copy()
        mass[0] = 0.0
    ncurr = int(indata.get_int("NCURR", 0))
    iotas = np.asarray(prof.get("iota", np.zeros((ns,), dtype=float)))
    if iotas.size:
        iotas = iotas.copy()
        iotas[0] = 0.0
    from .energy import _iotaf_from_iotas

    iotaf = np.asarray(_iotaf_from_iotas(iotas, lrfp=bool(indata.get_bool("LRFP", False))))

    # For current-driven runs (ncurr=1), VMEC updates iota from the force
    # balance (add_fluxes). Recompute iotas/iotaf from the final state to match
    # VMEC2000 even when callers supplied input-derived flux/profile payloads.
    if (
        ncurr == 1
        and os.getenv("VMEC_JAX_DISABLE_WOUT_NCURR_RECOMPUTE", "0") in ("", "0")
    ):
        chips, iotas, iotaf = equilibrium_iota_profiles_from_state(
            state=state,
            static=static,
            indata=indata,
            signgs=int(signgs),
        )
        chips = np.asarray(chips, dtype=float)
        iotas = np.asarray(iotas, dtype=float)
        iotaf = np.asarray(iotaf, dtype=float)
        chipf_wout = _chipf_from_chips(chips)

    # Geometry coefficients on the full mesh (convert internal -> external).
    m_arr = np.asarray(main_modes.m, dtype=int)
    n_arr = np.asarray(main_modes.n, dtype=int)
    sqrt2 = np.sqrt(2.0)
    mscale = np.where(m_arr == 0, 1.0, sqrt2)
    nscale = np.where(np.abs(n_arr) == 0, 1.0, sqrt2)
    mode_scale = (mscale * nscale)[None, :]
    lconm1 = bool(getattr(cfg, "lconm1", True))
    Rcos_use, Zsin_use, Rsin_use, Zcos_use = vmec_m1_internal_to_physical_signed_host(
        Rcos=np.asarray(state.Rcos, dtype=float),
        Zsin=np.asarray(state.Zsin, dtype=float),
        Rsin=np.asarray(state.Rsin, dtype=float),
        Zcos=np.asarray(state.Zcos, dtype=float),
        modes=main_modes,
        lthreed=bool(ntor > 0),
        lasym=bool(lasym),
        lconm1=bool(lconm1),
    )
    rmnc = np.asarray(Rcos_use, dtype=float) * mode_scale
    rmns = np.asarray(Rsin_use, dtype=float) * mode_scale
    zmnc = np.asarray(Zcos_use, dtype=float) * mode_scale
    zmns = np.asarray(Zsin_use, dtype=float) * mode_scale
    if not bool(lasym):
        rmns = np.zeros_like(rmnc)
        zmnc = np.zeros_like(zmns)
    lmnc_internal = np.asarray(state.Lcos, dtype=float)
    lmns_internal = np.asarray(state.Lsin, dtype=float)
    lmnc_internal = lmnc_internal * mode_scale
    lmns_internal = lmns_internal * mode_scale

    mnmax_nyq = int(nyq_modes.K)
    z2 = np.zeros((ns, mnmax_nyq), dtype=float)
    z1 = np.zeros((ns,), dtype=float)

    # Toroidal flux (VMEC `phi`) in physical units.
    phipf_internal = np.asarray(flux.phipf, dtype=float)
    phipf_out = phipf_internal * float(2.0 * np.pi * signgs)
    chipf_out = np.asarray(chipf_wout, dtype=float) * float(2.0 * np.pi * signgs)
    phi = np.asarray(cumrect_s_halfmesh(phipf_out, s))

    # Axis coefficients from m=0 modes on the magnetic axis.
    raxis_cc = np.zeros((ntor + 1,), dtype=float)
    raxis_cs = np.zeros_like(raxis_cc)
    zaxis_cs = np.zeros_like(raxis_cc)
    zaxis_cc = np.zeros_like(raxis_cc)
    for nval in range(ntor + 1):
        mask = (m_arr == 0) & (n_arr == nval)
        if np.any(mask):
            idx = int(np.where(mask)[0][0])
            raxis_cc[nval] = float(rmnc[0, idx])
            raxis_cs[nval] = float(rmns[0, idx])
            zaxis_cs[nval] = float(zmns[0, idx])
            zaxis_cc[nval] = float(zmnc[0, idx])

    Aminor_p = 0.0
    Rmajor_p = 0.0
    aspect = 0.0

    # Build VMEC parity grids for Nyquist outputs.

    class _WoutLike:
        __slots__ = (
            "phipf",
            "phips",
            "chipf",
            "iotaf",
            "iotas",
            "signgs",
            "nfp",
            "mpol",
            "ntor",
            "lasym",
            "flux_is_internal",
            "ncurr",
            "lcurrent",
            "icurv",
            "mass",
            "gamma",
        )

        def __init__(self):
            self.phipf = np.asarray(flux.phipf)
            self.phips = np.asarray(flux.phips)
            self.chipf = np.asarray(chipf_wout)
            self.iotaf = np.asarray(iotaf)
            self.iotas = np.asarray(iotas)
            self.signgs = int(signgs)
            self.nfp = int(nfp)
            self.mpol = int(mpol)
            self.ntor = int(ntor)
            self.lasym = bool(lasym)
            self.flux_is_internal = True
            self.ncurr = int(ncurr)
            self.lcurrent = bool(ncurr == 1)
            self.icurv = np.asarray(_icurv_full_mesh_from_indata(indata=indata, s_full=s, signgs=int(signgs)))
            self.mass = np.asarray(mass)
            self.gamma = float(gamma)

    wout_like = _WoutLike()
    from .vmec_forces import vmec_forces_rz_from_wout

    # Output diagnostics should default to the same IEQUI path used by the run.
    # A forced IEQUI=1 path is retained only as an explicit debug override.
    indata_wout = indata
    force_iequi1 = os.getenv("VMEC_JAX_WOUT_FORCE_IEQUI1", "0") not in ("", "0")
    if force_iequi1:
        try:
            indata_wout = InData(
                scalars=dict(indata.scalars),
                indexed=dict(indata.indexed),
                source_path=indata.source_path,
            )
            indata_wout.scalars["IEQUI"] = 1
        except Exception:
            indata_wout = indata

    k_force = None
    if wout_timing_enabled:
        t0 = _time.perf_counter()
    reuse_final_bcovar_env = os.getenv("VMEC_JAX_WOUT_REUSE_FINAL_BCOVAR", "").strip().lower()
    reuse_final_bcovar = reuse_final_bcovar_env not in ("", "0", "false", "no", "off")
    if force_payload_override is not None and (reuse_final_bcovar or not wout_fast_bcovar) and (not force_iequi1):
        k_force = force_payload_override
        if has_jax():
            try:
                k_force = jax.device_get(k_force)
            except Exception:
                pass
        bc = _bcovar_from_force_payload_with_geometry(geom, k_force)
    elif wout_fast_bcovar:
        with _numpy_module_patch():
            bc = vmec_bcovar_half_mesh_from_wout(
                state=state,
                static=static,
                wout=wout_like,
                pres=pres,
                use_wout_bsup=False,
                use_wout_bsub_for_lambda=False,
                use_wout_bmag_for_bsq=False,
                use_vmec_synthesis=True,
                trig=None,
            )
        if has_jax():
            try:
                bc = jax.device_get(bc)
            except Exception:
                pass
    else:
        wout_force_vmec_synth_env = os.getenv("VMEC_JAX_WOUT_FORCE_VMEC_SYNTH", "").strip().lower()
        wout_force_vmec_synth = wout_force_vmec_synth_env not in ("", "0", "false", "no")
        k_force = vmec_forces_rz_from_wout(
            state=state,
            static=static,
            wout=wout_like,
            indata=indata_wout,
            use_wout_bsup=False,
            use_vmec_synthesis=wout_force_vmec_synth,
            trig=None,
        )
        if has_jax():
            try:
                k_force = jax.device_get(k_force)
            except Exception:
                pass
        bc = _bcovar_from_force_payload_with_geometry(geom, k_force)
    if wout_timing_enabled:
        wout_timing["forces_bcovar_s"] = _time.perf_counter() - t0

    # VMEC output routes the bss/jxbforce path through the force-kernel arrays
    # (crmn_e/czmn_e and friends). In vmec_jax, the bcovar/Jacobian-based path
    # currently matches VMEC2000 bsubsmns more reliably, so keep force-kernels
    # opt-in.
    _force_bss_env = os.getenv("VMEC_JAX_WOUT_FORCE_BSS", "").strip().lower()
    if _force_bss_env == "":
        # Default to bcovar/Jacobian bss inputs. Force-kernel bss inputs remain
        # opt-in for targeted debugging.
        use_force_bss = False
    else:
        use_force_bss = _force_bss_env not in ("0", "false", "no")
    bsupu_bss = np.asarray(bc.bsupu, dtype=float)
    bsupv_bss = np.asarray(bc.bsupv, dtype=float)
    ru12_bss = None
    zu12_bss = None
    rs_bss = None
    zs_bss = None
    crmn_e_sym = None
    czmn_e_sym = None
    bzmn_e_sym = None
    brmn_e_sym = None
    azmn_e_sym = None
    armn_e_sym = None
    # For bss, default to using the half-mesh Jacobian (bcovar parity) and
    # avoid force-parity geometry unless explicitly requested.
    use_parity_geom_bss = os.getenv("VMEC_JAX_BSS_FROM_PARITY_GEOM", "1") not in ("", "0")
    geom_bss = geom if use_parity_geom_bss else {}

    def _force_sym(arr, kind: str):
        arr_np = np.asarray(arr, dtype=float)
        # For LASYM there is no symmetry reduction; keep full-grid values.
        if bool(lasym):
            return arr_np
        # Some call sites provide already-reduced theta grids; only apply the
        # full-grid symforce operation when the grid matches ntheta1.
        if int(arr_np.shape[1]) < int(trig.ntheta1):
            return arr_np
        # Enforce stellarator symmetry on the full grid when lasym=False.
        return _vmec_symforce_apply(f=arr_np, trig=trig, kind=kind)

    if use_force_bss and (k_force is None):
        wout_force_vmec_synth_env = os.getenv("VMEC_JAX_WOUT_FORCE_VMEC_SYNTH", "").strip().lower()
        wout_force_vmec_synth = wout_force_vmec_synth_env not in ("", "0", "false", "no")
        k_force = vmec_forces_rz_from_wout(
            state=state,
            static=static,
            wout=wout_like,
            indata=indata_wout,
            use_wout_bsup=False,
            use_vmec_synthesis=wout_force_vmec_synth,
            trig=None,
        )
        if has_jax():
            try:
                k_force = jax.device_get(k_force)
            except Exception:
                pass

    if use_force_bss and (k_force is not None):
        if hasattr(k_force, "crmn_e") and hasattr(k_force, "czmn_e"):
            crmn_e_sym = _force_sym(k_force.crmn_e, "crs")
            czmn_e_sym = _force_sym(k_force.czmn_e, "czs")
            bsupu_bss = crmn_e_sym
            bsupv_bss = czmn_e_sym
        if hasattr(k_force, "bzmn_e"):
            bzmn_e_sym = _force_sym(k_force.bzmn_e, "bzs")
            rs_bss = bzmn_e_sym
        if hasattr(k_force, "brmn_e"):
            brmn_e_sym = _force_sym(k_force.brmn_e, "brs")
            zs_bss = brmn_e_sym
        if hasattr(k_force, "azmn_e"):
            azmn_e_sym = _force_sym(k_force.azmn_e, "azs")
            ru12_bss = azmn_e_sym
        if hasattr(k_force, "armn_e"):
            armn_e_sym = _force_sym(k_force.armn_e, "ars")
            zu12_bss = armn_e_sym
    if os.getenv("VMEC_JAX_DUMP_BSUB_PARITY", "") not in ("", "0"):
        outdir = Path(os.getenv("VMEC_JAX_DUMP_DIR", ".")).expanduser().resolve()
        outdir.mkdir(parents=True, exist_ok=True)
        np.savez(
            outdir / "bsub_parity_dump.npz",
            s=np.asarray(s, dtype=float),
            bsubu=np.asarray(getattr(bc, "bsubu"), dtype=float),
            bsubv=np.asarray(getattr(bc, "bsubv"), dtype=float),
            bsubu_e=np.asarray(getattr(bc, "bsubu_e"), dtype=float),
            bsubv_e=np.asarray(getattr(bc, "bsubv_e"), dtype=float),
            bsubu_e_scaled=np.asarray(getattr(bc, "bsubu_e_scaled"), dtype=float),
            bsubv_e_scaled=np.asarray(getattr(bc, "bsubv_e_scaled"), dtype=float),
            bsubu_parity_even=np.asarray(getattr(bc, "bsubu_parity_even"), dtype=float),
            bsubu_parity_odd=np.asarray(getattr(bc, "bsubu_parity_odd"), dtype=float),
            bsubv_parity_even=np.asarray(getattr(bc, "bsubv_parity_even"), dtype=float),
            bsubv_parity_odd=np.asarray(getattr(bc, "bsubv_parity_odd"), dtype=float),
        )
    if os.getenv("VMEC_JAX_DUMP_BSUBH", "") not in ("", "0"):
        outdir = Path(os.getenv("VMEC_JAX_DUMP_DIR", ".")).expanduser().resolve()
        outdir.mkdir(parents=True, exist_ok=True)
        np.savez(
            outdir / "bsubh_wout.npz",
            s=np.asarray(s, dtype=float),
            bsupu=np.asarray(bsupu_bss, dtype=float),
            bsupv=np.asarray(bsupv_bss, dtype=float),
            bsubu=np.asarray(getattr(bc, "bsubu"), dtype=float),
            bsubv=np.asarray(getattr(bc, "bsubv"), dtype=float),
        )

    # Derived 1D profiles and scalars.
    norms = vmec_force_norms_from_bcovar_dynamic(bc=bc, trig=trig, s=s, signgs=int(signgs))
    if has_jax():
        try:
            norms = jax.device_get(norms)
        except Exception:
            pass
    vp = np.asarray(norms.vp, dtype=float)
    wb = float(np.asarray(norms.wb))
    wp = float(np.asarray(norms.wp))
    volume = float(np.asarray(norms.volume))
    volume_p = volume * float(4.0 * np.pi**2)
    betatotal = (wp / wb) if wb != 0.0 else 0.0

    # Reconstruct pressure from mass/vp to match VMEC's bcovar path.
    pres = np.zeros_like(vp)
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.where(vp != 0.0, vp, 1.0)
        pres = np.where(vp != 0.0, mass / (denom**gamma), 0.0)
    if pres.size:
        pres = pres.copy()
        pres[0] = 0.0
    if ns < 2:
        presf = pres.copy()
    else:
        presf = np.zeros_like(pres)
        if ns >= 3:
            presf[0] = 1.5 * pres[1] - 0.5 * pres[2]
        else:
            presf[0] = pres[1]
        presf[1:-1] = 0.5 * (pres[1:-1] + pres[2:])
        presf[-1] = 1.5 * pres[-1] - 0.5 * pres[-2]

    wint = _vmec_wint_from_trig(trig)
    Aminor_p, Rmajor_p, aspect, volume_p, _ = _compute_aspectratio(
        R=np.asarray(geom["R"]),
        Zu=np.asarray(geom["Zu"]),
        wint=wint,
    )

    # buco/bvco/jcur* will be computed after aligning bsubu sign for wout output.
    buco = bvco = jcuru = jcurv = None
    equif = None

    # Nyquist Fourier coefficients for fields stored in wout.
    bsupu_out = np.asarray(bc.bsupu)
    bsupv_out = np.asarray(bc.bsupv)
    if use_force_bss and (k_force is not None):
        if hasattr(k_force, "crmn_e") and hasattr(k_force, "czmn_e"):
            if crmn_e_sym is None:
                crmn_e_sym = _force_sym(k_force.crmn_e, "crs")
            if czmn_e_sym is None:
                czmn_e_sym = _force_sym(k_force.czmn_e, "czs")
            bsupu_out = crmn_e_sym
            bsupv_out = czmn_e_sym
    bsubu_out = np.asarray(bc.bsubu).copy()
    bsubv_out = np.asarray(bc.bsubv).copy()
    bsubu_raw = bsubu_out.copy()
    bsubv_raw = bsubv_out.copy()
    bsubv_lasym_asym_source = None
    bsubv_lasym_asym_filter_u = None
    if bool(lasym) and hasattr(bc, "bsubv_e"):
        # VMEC fileout forces IEQUI=1 before wrout. Diagnostics show only the
        # LASYM bsubv sine output channel follows this corrected half-mesh IEQUI
        # source; keep the existing raw channels for bsubvmnc/bsubu parity.
        bsubv_lasym_asym_source = _apply_bsubv_equif_correction(
            bsubv=np.asarray(getattr(bc, "bsubv"), dtype=float),
            bsubv_e=np.asarray(getattr(bc, "bsubv_e"), dtype=float),
            trig=trig,
        )
    if os.getenv("VMEC_JAX_DUMP_BSUB_SOURCES", "") not in ("", "0"):
        tag = os.getenv("VMEC_JAX_DUMP_TAG", "").strip()
        outdir = Path(os.getenv("VMEC_JAX_DUMP_DIR", ".")).expanduser().resolve()
        outdir.mkdir(parents=True, exist_ok=True)
        payload = {
            "bsubu": np.asarray(getattr(bc, "bsubu"), dtype=float),
            "bsubv": np.asarray(getattr(bc, "bsubv"), dtype=float),
        }
        for key in (
            "bsubu_e",
            "bsubv_e",
            "bsubu_e_scaled",
            "bsubv_e_scaled",
            "bsubu_preblend",
            "bsubv_preblend",
            "bsubu_parity_even",
            "bsubu_parity_odd",
            "bsubv_parity_even",
            "bsubv_parity_odd",
        ):
            val = getattr(bc, key, None)
            if val is not None:
                payload[key] = np.asarray(val, dtype=float)
        np.savez(outdir / f"bsub_sources{('_' + tag) if tag else ''}.npz", **payload)

    # VMEC wrout.f uses the *raw* bsubu/bsubv for Fourier output (bsubumnc/etc).
    # JXBFORCE-style diagnostics (jdotb/Mercier) use the equilibrated + filtered
    # fields. Keep both paths explicit to match VMEC output.
    bsubu_diag = bsubu_out
    bsubv_diag = bsubv_out
    bsub_src = os.getenv("VMEC_JAX_MERCIER_BSUB_SOURCE", "").strip().lower()
    if bsub_src in {"bsubu_e", "bsubu_e_scaled", "bsubu"}:
        u_name = bsub_src
        v_name = bsub_src.replace("bsubu", "bsubv")
        if hasattr(bc, u_name) and hasattr(bc, v_name):
            bsubu_diag = np.asarray(getattr(bc, u_name), dtype=float)
            bsubv_diag = np.asarray(getattr(bc, v_name), dtype=float)
    elif os.getenv("VMEC_JAX_MERCIER_USE_BSUBE", "0") not in ("", "0"):
        if hasattr(bc, "bsubu_e") and hasattr(bc, "bsubv_e"):
            bsubu_diag = np.asarray(getattr(bc, "bsubu_e"), dtype=float)
            bsubv_diag = np.asarray(getattr(bc, "bsubv_e"), dtype=float)
    bsubv_diag_from_raw = True
    # VMEC fileout.f forces IEQUI=1 before calling funct3d/wrout at output
    # time, regardless of the runtime IEQUI used in iterations.
    iequi = 1
    # VMEC parity: keep the raw bsubv path by default for Mercier/jdotb parity.
    disable_equif_corr = os.getenv("VMEC_JAX_DISABLE_BSUBV_EQUI_CORR", "1") not in ("", "0")
    if (iequi == 1) and (not disable_equif_corr) and getattr(bc, "bsubv_e", None) is not None:
        bsubv_diag = _apply_bsubv_equif_correction(
            bsubv=bsubv_diag,
            bsubv_e=np.asarray(bc.bsubv_e),
            trig=trig,
        )
        bsubv_diag_from_raw = False
    if wout_timing_enabled:
        t0 = _time.perf_counter()
    bsubs_half = _compute_bsubs_half_mesh(
        state=state,
        geom_modes=static.modes,
        s=np.asarray(s, dtype=float),
        lconm1=bool(getattr(cfg, "lconm1", True)),
        lthreed=bool(ntor > 0),
        lasym=bool(lasym),
        bsupu=bsupu_bss,
        bsupv=bsupv_bss,
        trig=trig,
        geom=geom_bss,
        jac_half=bc.jac,
        force_rs=rs_bss,
        force_zs=zs_bss,
        force_ru12=ru12_bss,
        force_zu12=zu12_bss,
        apply_scalxc=os.getenv("VMEC_JAX_BSS_APPLY_SCALXC", "1") not in ("", "0"),
    )
    if wout_timing_enabled:
        wout_timing["bsubs_half_s"] = _time.perf_counter() - t0
    # In VMEC's fileout path, jxbforce is called before wrout and updates bsubs
    # to a full-mesh representation via:
    #   bsubs(js) = 0.5*(bsubs(js) + bsubs(js+1)), js=2..ns-1
    # then endpoint extrapolation:
    #   bsubs(1)  = 2*bsubs(2)  - bsubs(3)
    #   bsubs(ns) = 2*bsubs(ns) - bsubs(ns-1)
    # wrout then transforms this updated array.
    bsubs_full = np.asarray(bsubs_half, dtype=float).copy()
    if ns > 0:
        # jxbforce initializes bsubs(1,:)=0 before full-mesh averaging.
        bsubs_full[0] = 0.0
    if ns > 2:
        bsubs_full[1:-1] = 0.5 * (bsubs_full[1:-1] + bsubs_full[2:])
        bsubs_full[0] = 2.0 * bsubs_full[1] - bsubs_full[2]
        bsubs_full[-1] = 2.0 * bsubs_full[-1] - bsubs_full[-2]

    # JXBFORCE applies a low-pass filter on bsubu/bsubv using (mpol-1, ntor).
    skip_bsub_filter = os.getenv("VMEC_JAX_SKIP_BSUB_FILTER", "") not in ("", "0")
    if wout_light:
        skip_bsub_filter = True
    filter_from_raw = os.getenv("VMEC_JAX_MERCIER_FILTER_FROM_RAW", "0") not in ("", "0")

    if wout_timing_enabled:
        t0 = _time.perf_counter()
    if bool(lasym):
        strict_lasym_loop = os.getenv("VMEC_JAX_WROUT_LASYM_STRICT", "") not in ("", "0", "false", "no")
        if strict_lasym_loop:
            use_lasym_loop = True
        else:
            # Default to vectorized LASYM Nyquist transforms. The explicit
            # loop path is retained for parity debugging via env toggle.
            use_lasym_loop = os.getenv("VMEC_JAX_WROUT_LASYM_LOOP", "0") not in ("", "0", "false", "no")
        if (not skip_bsub_filter) and (os.getenv("VMEC_JAX_WROUT_LASYM_FILTER", "1") not in ("", "0")):
            use_parity_channels = os.getenv("VMEC_JAX_LASYM_FILTER_USE_PARITY_CHANNELS", "0") not in (
                "",
                "0",
                "false",
                "no",
            )
            bsubu_even_filter = getattr(bc, "bsubu_parity_even", None) if use_parity_channels else None
            bsubu_odd_filter = getattr(bc, "bsubu_parity_odd", None) if use_parity_channels else None
            bsubv_even_filter = getattr(bc, "bsubv_parity_even", None) if use_parity_channels else None
            bsubv_odd_filter = getattr(bc, "bsubv_parity_odd", None) if use_parity_channels else None
            if bsubv_lasym_asym_source is not None:
                bsubv_lasym_asym_filter_u = np.asarray(bsubu_out, dtype=float).copy()
            bsubu_out, bsubv_out = _filter_bsubuv_jxbforce_lasym_loop(
                bsubu=np.asarray(bsubu_out, dtype=float),
                bsubv=np.asarray(bsubv_out, dtype=float),
                trig=trig,
                mmax_force=max(int(mpol) - 1, 0),
                nmax_force=int(ntor),
                s=np.asarray(s, dtype=float),
                bsubu_even=None if bsubu_even_filter is None else np.asarray(bsubu_even_filter, dtype=float),
                bsubu_odd=None if bsubu_odd_filter is None else np.asarray(bsubu_odd_filter, dtype=float),
                bsubv_even=None if bsubv_even_filter is None else np.asarray(bsubv_even_filter, dtype=float),
                bsubv_odd=None if bsubv_odd_filter is None else np.asarray(bsubv_odd_filter, dtype=float),
            )
            if bsubv_lasym_asym_source is not None:
                _, bsubv_lasym_asym_source = _filter_bsubuv_jxbforce_lasym_loop(
                    bsubu=np.asarray(bsubv_lasym_asym_filter_u, dtype=float),
                    bsubv=np.asarray(bsubv_lasym_asym_source, dtype=float),
                    trig=trig,
                    mmax_force=max(int(mpol) - 1, 0),
                    nmax_force=int(ntor),
                    s=np.asarray(s, dtype=float),
                    bsubu_even=None,
                    bsubu_odd=None,
                    bsubv_even=None,
                    bsubv_odd=None,
                )
            bsubu_diag = np.asarray(bsubu_out, dtype=float)
            bsubv_diag = np.asarray(bsubv_out, dtype=float)
        pres_h = np.asarray(pres, dtype=float)[:, None, None]
        bmag = np.sqrt(2.0 * np.abs(np.asarray(bc.bsq) - pres_h))
        sqrtg = np.asarray(bc.jac.sqrtg)

        if os.getenv("VMEC_JAX_DUMP_BSUB_PRE_SYM", "") not in ("", "0"):
            outdir = Path(os.getenv("VMEC_JAX_DUMP_DIR", ".")).expanduser().resolve()
            outdir.mkdir(parents=True, exist_ok=True)
            tag = os.getenv("VMEC_JAX_DUMP_TAG", "").strip()
            name = "bsub_pre_sym_jax" + (f"_{tag}" if tag else "") + ".dat"
            path = outdir / name
            bsubu_dump = np.asarray(bsubu_out, dtype=float)
            bsubv_dump = np.asarray(bsubv_out, dtype=float)
            bsupu_dump = np.asarray(bsupu_out, dtype=float)
            bsupv_dump = np.asarray(bsupv_out, dtype=float)
            bsubs_dump = np.asarray(bsubs_full, dtype=float)
            ns_d, ntheta_d, nzeta_d = bsubu_dump.shape
            with path.open("w") as f:
                f.write("# bsub pre-symoutput dump (full grid)\n")
                f.write(f"ns={ns_d}\n")
                f.write(f"ntheta3={ntheta_d}\n")
                f.write(f"nzeta={nzeta_d}\n")
                f.write("columns: js lt lz bsubu bsubv bsupu bsupv bsubs\n")
                for lt in range(ntheta_d):
                    for lz in range(nzeta_d):
                        for js in range(ns_d):
                            f.write(
                                f"{js + 1:6d}{lt + 1:6d}{lz + 1:6d}"
                                f"{bsubu_dump[js, lt, lz]:24.16E}"
                                f"{bsubv_dump[js, lt, lz]:24.16E}"
                                f"{bsupu_dump[js, lt, lz]:24.16E}"
                                f"{bsupv_dump[js, lt, lz]:24.16E}"
                                f"{bsubs_dump[js, lt, lz]:24.16E}\n"
                            )

        bsubu_sym, bsubu_asym = _vmec_symoutput_split(f=bsubu_out, trig=trig)
        bsubv_sym, bsubv_asym = _vmec_symoutput_split(f=bsubv_out, trig=trig)
        if bsubv_lasym_asym_source is not None:
            _, bsubv_asym = _vmec_symoutput_split(f=bsubv_lasym_asym_source, trig=trig)
        bsupu_sym, bsupu_asym = _vmec_symoutput_split(f=bsupu_out, trig=trig)
        bsupv_sym, bsupv_asym = _vmec_symoutput_split(f=bsupv_out, trig=trig)
        bsubs_sym, bsubs_asym = _vmec_symoutput_split(f=bsubs_full, trig=trig, reversed_sym=True)
        bmag_sym, bmag_asym = _vmec_symoutput_split(f=bmag, trig=trig)
        sqrtg_sym, sqrtg_asym = _vmec_symoutput_split(f=sqrtg, trig=trig)
        if use_lasym_loop:
            lasym_coeffs = _vmec_wrout_nyquist_lasym_loop(
                bsq=bmag_sym,
                gsqrt=sqrtg_sym,
                bsubu=bsubu_sym,
                bsubv=bsubv_sym,
                bsubs=bsubs_sym,
                bsupu=bsupu_sym,
                bsupv=bsupv_sym,
                modes=nyq_modes,
                trig=trig,
            )
            # Asymmetric channel: pass the asymmetric parts through the same loop
            # (wrout.f uses tsini for asym fields).
            lasym_coeffs_a = _vmec_wrout_nyquist_lasym_loop(
                bsq=bmag_asym,
                gsqrt=sqrtg_asym,
                bsubu=bsubu_asym,
                bsubv=bsubv_asym,
                bsubs=bsubs_asym,
                bsupu=bsupu_asym,
                bsupv=bsupv_asym,
                modes=nyq_modes,
                trig=trig,
            )
            gmnc = lasym_coeffs["gmnc"]
            bmnc = lasym_coeffs["bmnc"]
            bsubumnc = lasym_coeffs["bsubumnc"]
            bsubvmnc = lasym_coeffs["bsubvmnc"]
            bsubsmns = lasym_coeffs["bsubsmns"]
            bsupumnc = lasym_coeffs["bsupumnc"]
            bsupvmnc = lasym_coeffs["bsupvmnc"]

            gmns = lasym_coeffs_a["gmns"]
            bmns = lasym_coeffs_a["bmns"]
            bsubumns = lasym_coeffs_a["bsubumns"]
            bsubvmns = lasym_coeffs_a["bsubvmns"]
            bsubsmnc = lasym_coeffs_a["bsubsmnc"]
            bsupumns = lasym_coeffs_a["bsupumns"]
            bsupvmns = lasym_coeffs_a["bsupvmns"]
        else:
            gmnc = _vmec_wrout_nyquist_cos_coeffs(f=sqrtg_sym, modes=nyq_modes, trig=trig)
            bmnc = _vmec_wrout_nyquist_cos_coeffs(f=bmag_sym, modes=nyq_modes, trig=trig)
            bsubumnc = _vmec_wrout_nyquist_cos_coeffs(f=bsubu_sym, modes=nyq_modes, trig=trig)
            bsubvmnc = _vmec_wrout_nyquist_cos_coeffs(f=bsubv_sym, modes=nyq_modes, trig=trig)
            bsupumnc = _vmec_wrout_nyquist_cos_coeffs(f=bsupu_sym, modes=nyq_modes, trig=trig)
            bsupvmnc = _vmec_wrout_nyquist_cos_coeffs(f=bsupv_sym, modes=nyq_modes, trig=trig)
            bsubsmns = _vmec_wrout_nyquist_sin_coeffs(f=bsubs_sym, modes=nyq_modes, trig=trig)

            gmns = _vmec_wrout_nyquist_sin_coeffs(f=sqrtg_asym, modes=nyq_modes, trig=trig)
            bmns = _vmec_wrout_nyquist_sin_coeffs(f=bmag_asym, modes=nyq_modes, trig=trig)
            bsubumns = _vmec_wrout_nyquist_sin_coeffs(f=bsubu_asym, modes=nyq_modes, trig=trig)
            bsubvmns = _vmec_wrout_nyquist_sin_coeffs(f=bsubv_asym, modes=nyq_modes, trig=trig)
            bsupumns = _vmec_wrout_nyquist_sin_coeffs(f=bsupu_asym, modes=nyq_modes, trig=trig)
            bsupvmns = _vmec_wrout_nyquist_sin_coeffs(f=bsupv_asym, modes=nyq_modes, trig=trig)
            bsubsmnc = _vmec_wrout_nyquist_cos_coeffs(f=bsubs_asym, modes=nyq_modes, trig=trig)

        # VMEC output: bsubu/bsubv coefficients are band-limited to the base
        # spectral resolution (mpol-1, ntor). Higher Nyquist modes are zeroed
        # (wrout.f keeps them at ~0). Enforce this for LASYM parity.
        m_mask = np.asarray(nyq_modes.m, dtype=int)
        n_mask = np.asarray(nyq_modes.n, dtype=int)
        mask_bsub = (m_mask >= int(mpol)) | (np.abs(n_mask) > int(ntor))
        if np.any(mask_bsub):
            bsubumnc[:, mask_bsub] = 0.0
            bsubumns[:, mask_bsub] = 0.0
            bsubvmnc[:, mask_bsub] = 0.0
            bsubvmns[:, mask_bsub] = 0.0

        bsubumnc, bsubvmnc, bsubumns, bsubvmns = _vmec_wrout_lasym_bsubuv_output_scale(
            bsubumnc=bsubumnc,
            bsubvmnc=bsubvmnc,
            bsubumns=bsubumns,
            bsubvmns=bsubvmns,
        )

        if gmnc.shape[0] > 0:
            gmnc[0, :] = 0.0
            bmnc[0, :] = 0.0
            bsubumnc[0, :] = 0.0
            bsubvmnc[0, :] = 0.0
            bsupumnc[0, :] = 0.0
            bsupvmnc[0, :] = 0.0
        if gmns.shape[0] > 0:
            gmns[0, :] = 0.0
            bmns[0, :] = 0.0
            bsubumns[0, :] = 0.0
            bsubvmns[0, :] = 0.0
            bsupumns[0, :] = 0.0
            bsupvmns[0, :] = 0.0
        if (not use_lasym_loop) and (ns > 2):
            bsubsmns[0, :] = 2.0 * bsubsmns[1, :] - bsubsmns[2, :]
            bsubsmnc[0, :] = 2.0 * bsubsmnc[1, :] - bsubsmnc[2, :]
    else:
        gmnc = _vmec_wrout_nyquist_cos_coeffs(f=np.asarray(bc.jac.sqrtg), modes=nyq_modes, trig=trig)
        bsupu_out = np.asarray(bc.bsupu)
        bsupv_out = np.asarray(bc.bsupv)
        bsupumnc = _vmec_wrout_nyquist_cos_coeffs(f=bsupu_out, modes=nyq_modes, trig=trig)
        bsupvmnc = _vmec_wrout_nyquist_cos_coeffs(f=bsupv_out, modes=nyq_modes, trig=trig)
        bsubumnc = _vmec_wrout_nyquist_cos_coeffs(f=bsubu_out, modes=nyq_modes, trig=trig)
        bsubvmnc = _vmec_wrout_nyquist_cos_coeffs(f=bsubv_out, modes=nyq_modes, trig=trig)
        # Default to the vectorized path for performance; the loop-based path is
        # retained for parity debugging.
        use_loop = os.getenv("VMEC_JAX_WROUT_LOOP", "0") not in ("", "0")
        if use_loop:
            bsubsmns = _vmec_wrout_nyquist_sin_coeffs_loop(f=bsubs_full, modes=nyq_modes, trig=trig)
        else:
            bsubsmns = _vmec_wrout_nyquist_sin_coeffs(f=bsubs_full, modes=nyq_modes, trig=trig)

        pres_h = np.asarray(pres, dtype=float)[:, None, None]
        bmag = np.sqrt(2.0 * np.abs(np.asarray(bc.bsq) - pres_h))
        bmnc = _vmec_wrout_nyquist_cos_coeffs(f=bmag, modes=nyq_modes, trig=trig)

        # Axis values follow wrout.f (set to zero on js=1).
        if gmnc.shape[0] > 0:
            gmnc[0, :] = 0.0
            bsupumnc[0, :] = 0.0
            bsupvmnc[0, :] = 0.0
            bsubumnc[0, :] = 0.0
            bsubvmnc[0, :] = 0.0
            bmnc[0, :] = 0.0

        gmns = z2.copy()
        bsupumns = z2.copy()
        bsupvmns = z2.copy()
        bsubumns = z2.copy()
        bsubvmns = z2.copy()
        bsubsmnc = z2.copy()
        bmns = z2.copy()
        if ns > 2:
            bsubsmns[0, :] = 2.0 * bsubsmns[1, :] - bsubsmns[2, :]
    if wout_timing_enabled:
        wout_timing["nyquist_coeffs_s"] = _time.perf_counter() - t0
        t0 = _time.perf_counter()

    # Optional debug path: reconstruct physical real-space fields from the
    # Nyquist coefficients. This is *not* required for the default Mercier/jdotb
    # pipeline, which follows VMEC2000's jxbforce discretization directly on
    # the real-space (bsubu/bsubv) fields.
    bsubu_phys = None
    bsubv_phys = None
    if os.getenv("VMEC_JAX_MERCIER_USE_WROUT_BSUBUV", "") not in ("", "0"):
        from .fourier import build_helical_basis
        from .vmec_tomnsp import vmec_angle_grid

        grid_nyq = vmec_angle_grid(
            ntheta=int(cfg.ntheta),
            nzeta=int(cfg.nzeta),
            nfp=int(nfp),
            lasym=bool(lasym),
        )
        basis_nyq = build_helical_basis(nyq_modes, grid_nyq, cache=True)
        bsubu_phys = np.asarray(eval_fourier(bsubumnc, bsubumns, basis_nyq))
        bsubv_phys = np.asarray(eval_fourier(bsubvmnc, bsubvmns, basis_nyq))
    if wout_timing_enabled:
        t_bsub_filter = _time.perf_counter()
    if (not bool(lasym)) and (not skip_bsub_filter):
        if filter_from_raw:
            # Match jxbforce.f directly: filter from the real-space bsubu/bsubv
            # fields (with odd-m shalf handling done inside the loop filter).
            bsubu_diag, bsubv_diag = _filter_bsubuv_jxbforce_loop(
                bsubu=np.asarray(bsubu_diag, dtype=float),
                bsubv=np.asarray(bsubv_diag, dtype=float),
                trig=trig,
                mmax_force=max(int(mpol) - 1, 0),
                nmax_force=int(ntor),
                s=np.asarray(s, dtype=float),
            )
        else:
            # Match VMEC bcovar iequi=1 storage convention used by jxbforce:
            # parity channel 0 is the half-mesh field and parity channel 1 is
            # shalf*channel0 (before jxbforce divides odd by shalf).
            _psh = _pshalf_from_s(np.asarray(s, dtype=float))[:, None, None]
            if _psh.shape[0] > 1:
                _psh[0] = _psh[1]
            use_bc_parity = os.getenv("VMEC_JAX_BSUB_FILTER_USE_BC_PARITY", "0") not in ("", "0", "false", "no")
            if use_bc_parity and getattr(bc, "bsubu_parity_even", None) is not None:
                bsubu_even = np.asarray(getattr(bc, "bsubu_parity_even"), dtype=float)
                bsubv_even = np.asarray(getattr(bc, "bsubv_parity_even"), dtype=float)
                bsubu_odd = np.asarray(getattr(bc, "bsubu_parity_odd"), dtype=float)
                bsubv_odd = np.asarray(getattr(bc, "bsubv_parity_odd"), dtype=float)
            else:
                bsubu_even = np.asarray(bsubu_diag, dtype=float)
                bsubv_even = np.asarray(bsubv_diag, dtype=float)
                bsubu_odd = _psh * bsubu_even
                bsubv_odd = _psh * bsubv_even
            if os.getenv("VMEC_JAX_DUMP_BSUB_PARITY_INPUTS", "") not in ("", "0"):
                tag = os.getenv("VMEC_JAX_DUMP_TAG", "").strip()
                outdir = Path(os.getenv("VMEC_JAX_DUMP_DIR", ".")).expanduser().resolve()
                outdir.mkdir(parents=True, exist_ok=True)
                np.savez(
                    outdir / f"bsub_parity_inputs{('_' + tag) if tag else ''}.npz",
                    bsubu_diag=np.asarray(bsubu_diag, dtype=float),
                    bsubv_diag=np.asarray(bsubv_diag, dtype=float),
                    bsubu_even=np.asarray(bsubu_even, dtype=float),
                    bsubu_odd=np.asarray(bsubu_odd, dtype=float),
                    bsubv_even=np.asarray(bsubv_even, dtype=float),
                    bsubv_odd=np.asarray(bsubv_odd, dtype=float),
                    odd_needs_shalf=np.asarray(use_bc_parity, dtype=np.int32),
                )
            bsubu_diag, bsubv_diag = _filter_bsubuv_jxbforce_parity(
                bsubu_even=np.asarray(bsubu_even, dtype=float),
                bsubu_odd=np.asarray(bsubu_odd, dtype=float),
                bsubv_even=np.asarray(bsubv_even, dtype=float),
                bsubv_odd=np.asarray(bsubv_odd, dtype=float),
                trig=trig,
                mmax_force=max(int(mpol) - 1, 0),
                nmax_force=int(ntor),
                s=np.asarray(s, dtype=float),
            )
    if wout_timing_enabled:
        wout_timing["bsub_filter_s"] = _time.perf_counter() - t_bsub_filter
    # Match VMEC wrout: bsubu/bsubv Fourier output uses the jxbforce-filtered
    # fields, not the raw bcovar fields.
    bsubu_out = np.asarray(bsubu_diag, dtype=float)
    bsubv_out = np.asarray(bsubv_diag, dtype=float)
    if wout_timing_enabled:
        t_bsub_coeffs = _time.perf_counter()
    if not bool(lasym):
        bsubumnc = _vmec_wrout_nyquist_cos_coeffs(f=bsubu_out, modes=nyq_modes, trig=trig)
        bsubvmnc = _vmec_wrout_nyquist_cos_coeffs(f=bsubv_out, modes=nyq_modes, trig=trig)
        if bsubumnc.shape[0] > 0:
            bsubumnc[0, :] = 0.0
            bsubvmnc[0, :] = 0.0
    if wout_timing_enabled:
        wout_timing["bsub_coeffs_s"] = _time.perf_counter() - t_bsub_coeffs
    # Keep bsubsmns from the direct bsubs_half computation (wrout.f). The
    # Nyquist-reconstructed path is used only for consistency checks.

    if wout_light:
        buco = np.zeros((ns,), dtype=float)
        bvco = np.zeros((ns,), dtype=float)
        jcuru = np.zeros((ns,), dtype=float)
        jcurv = np.zeros((ns,), dtype=float)
        equif = np.zeros((ns,), dtype=float)
    else:
        if wout_timing_enabled:
            t_equif = _time.perf_counter()
        buco, bvco, jcuru, jcurv, equif = _compute_equif_wout(
            bsubu=bsubu_out,
            bsubv=bsubv_out,
            pres=pres,
            vp=vp,
            phipf=np.asarray(flux.phipf, dtype=float),
            chipf=np.asarray(chipf_wout, dtype=float),
            signgs=int(signgs),
            trig=trig,
            s=s,
        )
        if wout_timing_enabled:
            wout_timing["equif_s"] = _time.perf_counter() - t_equif

    # Current profile metadata for VMECPlot2.
    current_metadata = _wout_current_profile_metadata_from_indata(indata)
    ac = current_metadata.ac
    ac_aux_s = current_metadata.ac_aux_s
    ac_aux_f = current_metadata.ac_aux_f
    pcurr_type = current_metadata.pcurr_type
    piota_type = current_metadata.piota_type

    betapol = 0.0
    betator = 0.0
    betaxis = 0.0
    ctor = 0.0
    DMerc = np.zeros((ns,), dtype=float)
    Dshear = np.zeros((ns,), dtype=float)
    Dcurr = np.zeros((ns,), dtype=float)
    Dwell = np.zeros((ns,), dtype=float)
    Dgeod = np.zeros((ns,), dtype=float)
    D_R = np.zeros((ns,), dtype=float)
    H_glasser = np.zeros((ns,), dtype=float)
    glasser_correction = np.zeros((ns,), dtype=float)
    glasser_shear_valid = np.zeros((ns,), dtype=bool)
    jdotb = np.zeros((ns,), dtype=float)
    bdotb = np.zeros((ns,), dtype=float)
    bdotgradv = np.zeros((ns,), dtype=float)
    if not wout_light:
        try:
            # Mercier/jxbforce operate on real-space bsub* fields on VMEC's reduced
            # angular grid. Do not reconstruct from wout Fourier coefficients here:
            # wrout output transforms are not a strict inverse of eval_fourier.
            bsubu_merc = np.asarray(bsubu_diag, dtype=float)
            bsubv_merc = np.asarray(bsubv_diag, dtype=float)
            if os.getenv("VMEC_JAX_MERCIER_USE_RAW_BSUBUV", "") not in ("", "0"):
                bsubu_merc = np.asarray(bsubu_raw, dtype=float)
                bsubv_merc = np.asarray(bsubv_raw, dtype=float)
            elif os.getenv("VMEC_JAX_MERCIER_USE_WROUT_BSUBUV", "") not in ("", "0"):
                # Optional parity path for debugging: use Nyquist-reconstructed
                # bsubu/bsubv from wrout coefficients in the jxbforce scalars.
                bsubu_merc = np.asarray(bsubu_phys, dtype=float)
                bsubv_merc = np.asarray(bsubv_phys, dtype=float)
            elif os.getenv("VMEC_JAX_MERCIER_USE_RAW_BSUBV", "") not in ("", "0"):
                bsubv_merc = np.asarray(bsubv_raw, dtype=float)
            wint = _vmec_wint_from_trig(trig)
            if wout_timing_enabled:
                t_beta = _time.perf_counter()
            betaxis = _compute_eqfor_betaxis(
                pres=np.asarray(pres, dtype=float),
                vp=np.asarray(vp, dtype=float),
                bsq=np.asarray(bc.bsq, dtype=float),
                sqrtg=np.asarray(bc.jac.sqrtg, dtype=float),
                wint=wint,
                signgs=int(signgs),
            )
            betapol, betator, betatot_eq, betaxis = _compute_eqfor_beta(
                pres=np.asarray(pres, dtype=float),
                vp=np.asarray(vp, dtype=float),
                bsq=np.asarray(bc.bsq, dtype=float),
                r12=np.asarray(bc.jac.r12, dtype=float),
                bsupv=np.asarray(bc.bsupv, dtype=float),
                sqrtg=np.asarray(bc.jac.sqrtg, dtype=float),
                wint=wint,
                signgs=int(signgs),
            )
            if wout_timing_enabled:
                wout_timing["beta_s"] = _time.perf_counter() - t_beta
            betatotal = float(betatot_eq)
            ctor = _compute_ctor_from_buco(buco=np.asarray(buco, dtype=float), signgs=int(signgs), indata=indata)
            if wout_timing_enabled:
                t_mercier = _time.perf_counter()
            (
                DMerc,
                Dshear,
                Dcurr,
                Dwell,
                Dgeod,
                jdotb,
                bdotb,
                bdotgradv,
            ) = _compute_mercier(
                state=state,
                geom_modes=static.modes,
                s=np.asarray(s, dtype=float),
                lconm1=bool(getattr(cfg, "lconm1", True)),
                lthreed=bool(ntor > 0),
                lasym=bool(lasym),
                nfp=int(nfp),
                lbsubs=bool(lbsubs),
                mmax_force=max(int(mpol) - 1, 0),
                nmax_force=int(ntor),
                pres=np.asarray(pres, dtype=float),
                vp=np.asarray(vp, dtype=float),
                phips=np.asarray(flux.phips, dtype=float),
                iotas=np.asarray(iotas, dtype=float),
                # Use bcovar real-space fields directly (VMEC internal grid/normalization)
                bsq=np.asarray(bc.bsq, dtype=float),
                sqrtg=np.asarray(bc.jac.sqrtg, dtype=float),
                bsubu=bsubu_merc,
                bsubv=bsubv_merc,
                bsubu_parity_even=(
                    None
                    if getattr(bc, "bsubu_parity_even", None) is None
                    else np.asarray(getattr(bc, "bsubu_parity_even"), dtype=float)
                ),
                bsubu_parity_odd=(
                    None
                    if getattr(bc, "bsubu_parity_odd", None) is None
                    else np.asarray(getattr(bc, "bsubu_parity_odd"), dtype=float)
                ),
                bsubv_parity_even=(
                    None
                    if getattr(bc, "bsubv_parity_even", None) is None
                    else np.asarray(getattr(bc, "bsubv_parity_even"), dtype=float)
                ),
                bsubv_parity_odd=(
                    None
                    if getattr(bc, "bsubv_parity_odd", None) is None
                    else np.asarray(getattr(bc, "bsubv_parity_odd"), dtype=float)
                ),
                bsupu=np.asarray(bsupu_bss, dtype=float),
                bsupv=np.asarray(bsupv_bss, dtype=float),
                trig=trig,
                geom=geom_bss,
                jac_half=bc.jac,
                force_rs=rs_bss,
                force_zs=zs_bss,
                force_ru12=ru12_bss,
                force_zu12=zu12_bss,
                bsubu_raw=np.asarray(bsubu_raw, dtype=float),
                bsubv_raw=np.asarray(bsubv_raw, dtype=float),
                signgs=int(signgs),
            )
            D_R, H_glasser, glasser_correction, glasser_shear_valid = _glasser_from_wout_mercier_terms(
                DMerc=DMerc,
                Dshear=Dshear,
                Dcurr=Dcurr,
            )
            if wout_timing_enabled:
                wout_timing["mercier_s"] = _time.perf_counter() - t_mercier
        except Exception:
            if os.getenv("VMEC_JAX_STRICT_WOUT_DIAGNOSTICS", "") not in ("", "0"):
                raise

    # vmec_jax writes VMEC++-style diagnostic wout files for non-converged runs:
    # preserve every field computed from the last state and mark solver status.
    # Keep an explicit legacy escape hatch for tests/comparisons that require
    # VMEC2000's non-converged beta zeroing behavior.
    zero_beta_nonconv = os.getenv("VMEC_JAX_WOUT_ZERO_NONCONVERGED_BETA", "").strip().lower() not in (
        "",
        "0",
        "false",
        "no",
        "off",
    )
    if (not bool(converged)) and bool(zero_beta_nonconv):
        betatotal = 0.0
        betapol = 0.0
        betator = 0.0

    # Convert internal lambda coefficients to VMEC wout convention.
    from .field import lamscale_from_phips

    lamscale = float(np.asarray(lamscale_from_phips(flux.phips, s)))

    if wout_timing_enabled:
        wout_timing["jxbforce_mercier_s"] = _time.perf_counter() - t0

    lmns = _lambda_wout_from_full_mesh(
        lam_full=lmns_internal,
        m_modes=np.asarray(main_modes.m, dtype=int),
        s=s,
        phipf_internal=phipf_internal,
        lamscale=lamscale,
    )
    lmnc = _lambda_wout_from_full_mesh(
        lam_full=lmnc_internal,
        m_modes=np.asarray(main_modes.m, dtype=int),
        s=s,
        phipf_internal=phipf_internal,
        lamscale=lamscale,
    )
    if fsqt is None:
        fsqt_out = np.zeros((100,), dtype=float)
    else:
        fsqt_out = np.asarray(fsqt, dtype=float)

    if os.getenv("VMEC_JAX_DUMP_WROUT_MODES", "") not in ("", "0"):
        dump_dir = Path(os.getenv("VMEC_JAX_DUMP_DIR", ".")).expanduser().resolve()
        dump_dir.mkdir(parents=True, exist_ok=True)
        dump_path = dump_dir / "wrout_modes_jax.dat"
        m_modes = np.asarray(nyq_modes.m, dtype=int)
        n_modes = np.asarray(nyq_modes.n, dtype=int)
        gmnc_np = np.asarray(gmnc, dtype=float)
        gmns_np = np.asarray(gmns, dtype=float)
        bmnc_np = np.asarray(bmnc, dtype=float)
        bmns_np = np.asarray(bmns, dtype=float)
        bsubumnc_np = np.asarray(bsubumnc, dtype=float)
        bsubumns_np = np.asarray(bsubumns, dtype=float)
        bsubvmnc_np = np.asarray(bsubvmnc, dtype=float)
        bsubvmns_np = np.asarray(bsubvmns, dtype=float)
        bsubsmnc_np = np.asarray(bsubsmnc, dtype=float)
        bsubsmns_np = np.asarray(bsubsmns, dtype=float)
        bsupumnc_np = np.asarray(bsupumnc, dtype=float)
        bsupumns_np = np.asarray(bsupumns, dtype=float)
        bsupvmnc_np = np.asarray(bsupvmnc, dtype=float)
        bsupvmns_np = np.asarray(bsupvmns, dtype=float)
        with dump_path.open("w") as f:
            f.write("# wrout Fourier-mode dump (vmec_jax)\n")
            f.write(f"ns={ns}\n")
            f.write(f"mnmax_nyq={m_modes.size}\n")
            f.write("cols: js mn m n\n")
            f.write(" gmnc gmns bmnc bmns\n")
            f.write(" bsubumnc bsubumns bsubvmnc bsubvmns\n")
            f.write(" bsubsmnc bsubsmns\n")
            f.write(" bsupumnc bsupumns bsupvmnc bsupvmns\n")
            for js_idx in range(ns):
                for mn_idx in range(m_modes.size):
                    f.write(
                        f"{js_idx + 1:6d}{mn_idx + 1:6d}{int(m_modes[mn_idx]):6d}{int(n_modes[mn_idx]):6d}"
                        f"{gmnc_np[js_idx, mn_idx]:24.16E}{gmns_np[js_idx, mn_idx]:24.16E}"
                        f"{bmnc_np[js_idx, mn_idx]:24.16E}{bmns_np[js_idx, mn_idx]:24.16E}"
                        f"{bsubumnc_np[js_idx, mn_idx]:24.16E}{bsubumns_np[js_idx, mn_idx]:24.16E}"
                        f"{bsubvmnc_np[js_idx, mn_idx]:24.16E}{bsubvmns_np[js_idx, mn_idx]:24.16E}"
                        f"{bsubsmnc_np[js_idx, mn_idx]:24.16E}{bsubsmns_np[js_idx, mn_idx]:24.16E}"
                        f"{bsupumnc_np[js_idx, mn_idx]:24.16E}{bsupumns_np[js_idx, mn_idx]:24.16E}"
                        f"{bsupvmnc_np[js_idx, mn_idx]:24.16E}{bsupvmns_np[js_idx, mn_idx]:24.16E}\n"
                    )

    wout = WoutData(
        path=Path(path),
        ns=ns,
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        lasym=lasym,
        signgs=int(signgs),
        mnmax=int(main_modes.K),
        mpol_nyq=int(np.max(nyq_modes.m)) if int(nyq_modes.K) > 0 else 0,
        ntor_nyq=int(np.max(np.abs(nyq_modes.n))) if int(nyq_modes.K) > 0 else 0,
        mnmax_nyq=int(nyq_modes.K),
        xm=np.asarray(main_modes.m, dtype=int),
        xn=np.asarray(main_modes.n * nfp, dtype=int),
        xm_nyq=np.asarray(nyq_modes.m, dtype=int),
        xn_nyq=np.asarray(nyq_modes.n * nfp, dtype=int),
        rmnc=rmnc,
        rmns=rmns,
        zmnc=zmnc,
        zmns=zmns,
        lmnc=lmnc,
        lmns=lmns,
        phipf=phipf_out,
        chipf=chipf_out,
        phips=np.asarray(flux.phips, dtype=float),
        iotaf=np.asarray(iotaf, dtype=float),
        iotas=np.asarray(iotas, dtype=float),
        gmnc=np.asarray(gmnc, dtype=float),
        gmns=np.asarray(gmns, dtype=float),
        bsupumnc=np.asarray(bsupumnc, dtype=float),
        bsupumns=np.asarray(bsupumns, dtype=float),
        bsupvmnc=np.asarray(bsupvmnc, dtype=float),
        bsupvmns=np.asarray(bsupvmns, dtype=float),
        bsubumnc=np.asarray(bsubumnc, dtype=float),
        bsubumns=np.asarray(bsubumns, dtype=float),
        bsubvmnc=np.asarray(bsubvmnc, dtype=float),
        bsubvmns=np.asarray(bsubvmns, dtype=float),
        bsubsmns=np.asarray(bsubsmns, dtype=float),
        bsubsmnc=np.asarray(bsubsmnc, dtype=float),
        bmnc=np.asarray(bmnc, dtype=float),
        bmns=np.asarray(bmns, dtype=float),
        wb=float(wb),
        volume_p=float(volume_p),
        gamma=float(getattr(indata, "get_float", lambda *_: 0.0)("GAMMA", 0.0)),
        wp=float(wp),
        vp=np.asarray(vp, dtype=float),
        pres=np.asarray(pres, dtype=float),
        presf=np.asarray(presf, dtype=float),
        fsqr=float(fsqr),
        fsqz=float(fsqz),
        fsql=float(fsql),
        fsqt=np.asarray(fsqt_out, dtype=float),
        equif=np.asarray(equif, dtype=float),
        phi=np.asarray(phi, dtype=float),
        buco=np.asarray(buco, dtype=float),
        bvco=np.asarray(bvco, dtype=float),
        jcuru=np.asarray(jcuru, dtype=float),
        jcurv=np.asarray(jcurv, dtype=float),
        raxis_cc=np.asarray(raxis_cc, dtype=float),
        zaxis_cs=np.asarray(zaxis_cs, dtype=float),
        raxis_cs=np.asarray(raxis_cs, dtype=float),
        zaxis_cc=np.asarray(zaxis_cc, dtype=float),
        Aminor_p=float(Aminor_p),
        Rmajor_p=float(Rmajor_p),
        aspect=float(aspect),
        betatotal=float(betatotal),
        betapol=float(betapol),
        betator=float(betator),
        betaxis=float(betaxis),
        ctor=float(ctor),
        DMerc=np.asarray(DMerc, dtype=float),
        Dshear=np.asarray(Dshear, dtype=float),
        Dwell=np.asarray(Dwell, dtype=float),
        Dcurr=np.asarray(Dcurr, dtype=float),
        Dgeod=np.asarray(Dgeod, dtype=float),
        D_R=np.asarray(D_R, dtype=float),
        H=np.asarray(H_glasser, dtype=float),
        glasser_correction=np.asarray(glasser_correction, dtype=float),
        glasser_shear_valid=np.asarray(glasser_shear_valid, dtype=bool),
        jdotb=np.asarray(jdotb, dtype=float),
        bdotb=np.asarray(bdotb, dtype=float),
        bdotgradv=np.asarray(bdotgradv, dtype=float),
        ac=np.asarray(ac, dtype=float),
        ac_aux_s=np.asarray(ac_aux_s, dtype=float),
        ac_aux_f=np.asarray(ac_aux_f, dtype=float),
        pcurr_type=str(pcurr_type),
        piota_type=str(piota_type),
        ier_flag=0 if bool(converged) else 1,
        vmec_jax_converged=bool(converged),
        vmec_jax_status="converged" if bool(converged) else "nonconverged",
    )

    if wout_timing_enabled:
        wout_timing["total_s"] = _time.perf_counter() - t_wout_total_start
        try:
            parts = []
            for k in (
                "total_s",
                "trig_tables_s",
                "geom_synthesis_s",
                "forces_bcovar_s",
                "bsubs_half_s",
                "nyquist_coeffs_s",
                "equif_s",
                "beta_s",
                "mercier_s",
                "bsub_filter_s",
                "bsub_coeffs_s",
                "jxbforce_mercier_s",
            ):
                if k in wout_timing:
                    parts.append(f"{k}={wout_timing[k]:.3e}")
            print("[vmec_jax wout timing] " + " ".join(parts), flush=True)
        except Exception:
            pass
    return wout


def state_from_wout(wout: WoutData) -> VMECState:
    """Build a :class:`~vmec_jax.state.VMECState` from `wout` Fourier coefficients.

    Notes
    -----
    VMEC's ``wout`` files do **not** store the internal lambda coefficients in the
    same units VMEC uses in ``bcovar`` / ``totzsps``.

    In ``wrout.f`` VMEC writes (schematically, for each radial surface ``js``)::

        lmns_wout(:,js) = (lmns_internal(:,js) / phipf(js)) * lamscale

    to preserve an older output convention.

    For parity-style kernels that re-use VMEC's ``bcovar`` formulas, we therefore
    invert this scaling when constructing the state:

        lmns_internal = lmns_wout * phipf / lamscale
    """
    assert_main_modes_match_wout(wout=wout)
    layout = StateLayout(ns=wout.ns, K=int(wout.xm.size), lasym=bool(wout.lasym))

    # Reconstruct VMEC's internal lambda coefficients from the `wout` convention.
    # See `VMEC2000/Sources/Input_Output/wrout.f` (comment: "IF B^v ~ phip + lamu,
    # MUST DIVIDE BY phipf(js) below to maintain old-style format").
    from .field import lamscale_from_phips

    ns = int(wout.ns)
    if ns < 2:
        s = np.asarray([0.0], dtype=float)
    else:
        s = np.linspace(0.0, 1.0, ns, dtype=float)
    lamscale = float(np.asarray(lamscale_from_phips(wout.phips, s)))
    # VMEC's `wout` stores phipf scaled by 2π*signgs. Internally, lambda scaling
    # uses the unscaled phipf (= phipf_internal). Align the reconstruction with
    # bcovar's bsupv formula by undoing the 2π*signgs factor here.
    scale = float(2.0 * np.pi * float(getattr(wout, "signgs", 1)))
    phipf_internal = (
        np.asarray(wout.phipf, dtype=float) / scale if scale != 0.0 else np.asarray(wout.phipf, dtype=float)
    )
    if lamscale == 0.0:
        lam_scale = np.zeros((ns,), dtype=float)
    else:
        lam_scale = phipf_internal / lamscale  # (ns,)

    # VMEC writes lambda in a backward-compatible *half-mesh* convention (wrout.f),
    # which is not the internal full-mesh representation used by `totzsps`/`bcovar`.
    # We reproduce VMEC's own recovery logic from `load_xc_from_wout.f`:
    #   - undo the half-mesh interpolation (recurrence in `js`)
    #   - multiply by `phipf(js)` (undo old-style division)
    #   - divide by `lamscale` (undo old-style multiply)
    #
    # This yields lambda coefficients that are consistent with VMEC's internal
    # `bcovar` formulas when used with our `lamscale` scaling.
    lmns_full = _lambda_full_from_wout_half_mesh(
        lam_wout=np.asarray(wout.lmns),
        m_modes=np.asarray(wout.xm),
        s=s,
        phipf_internal=np.asarray(phipf_internal),
        lamscale=lamscale,
    )
    lmnc_full = _lambda_full_from_wout_half_mesh(
        lam_wout=np.asarray(wout.lmnc),
        m_modes=np.asarray(wout.xm),
        s=s,
        phipf_internal=np.asarray(phipf_internal),
        lamscale=lamscale,
    )

    m_arr = np.asarray(wout.xm, dtype=int)
    n_arr = (np.asarray(wout.xn, dtype=int) // int(wout.nfp)).astype(int)
    sqrt2 = np.sqrt(2.0)
    mscale = np.where(m_arr == 0, 1.0, sqrt2)
    nscale = np.where(np.abs(n_arr) == 0, 1.0, sqrt2)
    mode_scale = (1.0 / (mscale * nscale))[None, :]

    Rcos = np.asarray(wout.rmnc) * mode_scale
    Rsin = np.asarray(wout.rmns) * mode_scale
    Zcos = np.asarray(wout.zmnc) * mode_scale
    Zsin = np.asarray(wout.zmns) * mode_scale

    modes = vmec_mode_table(wout.mpol, wout.ntor)
    lthreed = bool(int(wout.ntor) > 0)
    lasym = bool(wout.lasym)
    lconm1 = bool(lthreed or lasym)
    Rcos, Zsin, Rsin, Zcos = vmec_m1_physical_to_internal_signed(
        Rcos=Rcos,
        Zsin=Zsin,
        Rsin=Rsin,
        Zcos=Zcos,
        modes=modes,
        lthreed=lthreed,
        lasym=lasym,
        lconm1=lconm1,
    )

    return VMECState(
        layout=layout,
        Rcos=Rcos,
        Rsin=Rsin,
        Zcos=Zcos,
        Zsin=Zsin,
        Lcos=np.asarray(lmnc_full) * mode_scale,
        Lsin=np.asarray(lmns_full) * mode_scale,
    )
