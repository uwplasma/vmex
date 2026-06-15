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
    vmec_realspace_synthesis_dzeta_phys,
    vmec_realspace_geom_from_state,
)
from .vmec_residue import vmec_pwint_from_trig
from .wout_schema import WoutData, _bool_from_nc, _nc_scalar, assert_main_modes_match_wout
from .wout_io import (
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
from . import wout_diagnostics as _wout_diagnostics
from . import wout_flux_helpers as _wout_flux_helpers
from . import wout_parity_helpers as _wout_parity_helpers
from .vmec_tomnsp import vmec_trig_tables


MU0 = 4e-7 * np.pi  # N/A^2
_chipf_from_chips = _wout_flux_helpers.chipf_from_chips
_compute_aspectratio = _wout_diagnostics.compute_aspectratio
_compute_ctor_from_buco = _wout_diagnostics.compute_ctor_from_buco
_compute_eqfor_beta = _wout_diagnostics.compute_eqfor_beta
_compute_eqfor_betaxis = _wout_diagnostics.compute_eqfor_betaxis
_glasser_from_wout_mercier_terms = _wout_diagnostics.glasser_from_wout_mercier_terms
_glasser_profiles_from_wout_variables = _wout_diagnostics.glasser_profiles_from_wout_variables
_icurv_full_mesh_from_indata = _wout_flux_helpers.icurv_full_mesh_from_indata
_lambda_full_from_wout_half_mesh = _wout_flux_helpers.lambda_full_from_wout_half_mesh
_lambda_half_mesh_weights = _wout_diagnostics.lambda_half_mesh_weights
_lambda_wout_from_full_mesh = _wout_flux_helpers.lambda_wout_from_full_mesh
_pshalf_from_s = _wout_diagnostics.pshalf_from_s
_safe_divide = _wout_diagnostics.safe_divide
_wout_phi_profile_from_variables = _wout_flux_helpers.wout_phi_profile_from_variables
_read_wout_scalar_metadata = read_wout_scalar_metadata
_bss_scalxc_undo_factor = _wout_parity_helpers.bss_scalxc_undo_factor
_bss_should_undo_scalxc = _wout_parity_helpers.bss_should_undo_scalxc
_undo_bss_scalxc_if_enabled = _wout_parity_helpers.undo_bss_scalxc_if_enabled


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


def _jxbforce_nyquist_limits(trig) -> tuple[int, int]:
    """Return VMEC jxbforce Nyquist cutoffs from grid sizes.

    In VMEC2000, ``mnyq`` / ``nnyq`` are geometric Nyquist limits from
    ``fixaray`` (based on ``ntheta2`` / ``nzeta``), not simply the maximum
    retained Fourier mode in a truncated transform loop.
    """
    ntheta2 = int(getattr(trig, "ntheta2", 0))
    cosnv = np.asarray(getattr(trig, "cosnv"))
    nzeta = int(cosnv.shape[0]) if cosnv.ndim >= 1 else 0
    mnyq = max(ntheta2 - 1, 0)
    nnyq = max(nzeta // 2, 0)
    return mnyq, nnyq


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
    from .solve_profile_helpers import _half_mesh_from_full_mesh, _mass_half_mesh_from_indata
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
    overg = jnp.where(sqrtg != 0.0, 1.0 / sqrtg, 0.0)
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
    chips = jnp.where(bot != 0.0, top / bot, jnp.zeros_like(top))
    if chips.shape[0] >= 1:
        chips = chips.at[0].set(0.0)
    iotas = jnp.where(phips != 0.0, chips / phips, jnp.zeros_like(chips))
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


def _compute_bsubs_half_mesh(
    *,
    state: VMECState,
    geom_modes,
    s: np.ndarray,
    lconm1: bool,
    lthreed: bool,
    lasym: bool,
    bsupu: np.ndarray,
    bsupv: np.ndarray,
    trig,
    geom: dict[str, Any],
    jac_half: Any | None = None,
    force_rs: np.ndarray | None = None,
    force_zs: np.ndarray | None = None,
    force_ru12: np.ndarray | None = None,
    force_zu12: np.ndarray | None = None,
    apply_m1_constraint: bool = False,
    apply_scalxc: bool = False,
) -> np.ndarray:
    """Compute bsubs on the half mesh using VMEC's bss.f conventions."""
    if bool(lasym):
        # LASYM path uses full-interval grids. When force-kernel parity is
        # supplied (via `geom["pr*"]` populated from symforce), the same
        # algebra as the VMEC bss.f half-mesh update applies.
        pass

    # Geometry fields split into even/odd-m components on the full mesh.
    # VMEC's realspace arrays are built directly from internal coefficients
    # (which already include the 1/sqrt(s) odd-m scaling), so we keep
    # apply_scalxc=False to match bss.f inputs.
    from .vmec_realspace import (
        vmec_realspace_synthesis,
        vmec_realspace_synthesis_dtheta,
    )

    m = np.asarray(geom_modes.m, dtype=int)
    mask_even = (m % 2) == 0
    mask_m1 = m == 1
    mask_odd_rest = (m % 2 == 1) & (~mask_m1)
    Rcos = np.asarray(state.Rcos)
    Rsin = np.asarray(state.Rsin)
    Zcos = np.asarray(state.Zcos)
    Zsin = np.asarray(state.Zsin)
    if bool(lconm1) and bool(apply_m1_constraint):
        Rcos, Zsin, Rsin, Zcos = vmec_m1_internal_to_physical_signed(
            Rcos=Rcos,
            Zsin=Zsin,
            Rsin=Rsin,
            Zcos=Zcos,
            modes=geom_modes,
            lthreed=bool(lthreed),
            lasym=bool(lasym),
            lconm1=bool(lconm1),
        )
    from .vmec_jacobian import _apply_vmec_axis_rules

    Rcos = _apply_vmec_axis_rules(Rcos, m)
    Rsin = _apply_vmec_axis_rules(Rsin, m)
    Zcos = _apply_vmec_axis_rules(Zcos, m)
    Zsin = _apply_vmec_axis_rules(Zsin, m)

    coeff_cos_stack = np.stack([Rcos, Zcos], axis=0)
    coeff_sin_stack = np.stack([Rsin, Zsin], axis=0)

    mask_even_f = mask_even.astype(float)
    mask_m1_f = mask_m1.astype(float)
    mask_odd_rest_f = mask_odd_rest.astype(float)

    def _eval_mask(mask: np.ndarray, *, deriv: str, apply_scalxc_local: bool):
        coeff_cos = coeff_cos_stack * mask[None, None, :]
        coeff_sin = coeff_sin_stack * mask[None, None, :]
        if deriv == "base":
            return vmec_realspace_synthesis(
                coeff_cos=coeff_cos,
                coeff_sin=coeff_sin,
                modes=geom_modes,
                trig=trig,
                coeffs_internal=True,
                apply_scalxc=bool(apply_scalxc_local),
                s=s,
            )
        if deriv == "dtheta":
            return vmec_realspace_synthesis_dtheta(
                coeff_cos=coeff_cos,
                coeff_sin=coeff_sin,
                modes=geom_modes,
                trig=trig,
                coeffs_internal=True,
                apply_scalxc=bool(apply_scalxc_local),
                s=s,
            )
        if deriv == "dzeta":
            return vmec_realspace_synthesis_dzeta_phys(
                coeff_cos=coeff_cos,
                coeff_sin=coeff_sin,
                modes=geom_modes,
                trig=trig,
                coeffs_internal=True,
                apply_scalxc=bool(apply_scalxc_local),
                s=s,
            )
        raise ValueError(f"Unknown deriv {deriv}")

    if bool(lasym):
        # LASYM: VMEC's even/odd realspace fields correspond to cos/sin phase
        # components (theta parity), not m-parity splits. Build them directly
        # from the cos/sin coefficient stacks.
        zeros = np.zeros_like(coeff_cos_stack)
        even_base = np.asarray(
            vmec_realspace_synthesis(
                coeff_cos=coeff_cos_stack,
                coeff_sin=zeros,
                modes=geom_modes,
                trig=trig,
                coeffs_internal=True,
                apply_scalxc=False,
                s=s,
            )
        )
        even_t = np.asarray(
            vmec_realspace_synthesis_dtheta(
                coeff_cos=coeff_cos_stack,
                coeff_sin=zeros,
                modes=geom_modes,
                trig=trig,
                coeffs_internal=True,
                apply_scalxc=False,
                s=s,
            )
        )
        even_p = np.asarray(
            vmec_realspace_synthesis_dzeta_phys(
                coeff_cos=coeff_cos_stack,
                coeff_sin=zeros,
                modes=geom_modes,
                trig=trig,
                coeffs_internal=True,
                apply_scalxc=False,
                s=s,
            )
        )
        odd_base = np.asarray(
            vmec_realspace_synthesis(
                coeff_cos=zeros,
                coeff_sin=coeff_sin_stack,
                modes=geom_modes,
                trig=trig,
                coeffs_internal=True,
                apply_scalxc=False,
                s=s,
            )
        )
        odd_t = np.asarray(
            vmec_realspace_synthesis_dtheta(
                coeff_cos=zeros,
                coeff_sin=coeff_sin_stack,
                modes=geom_modes,
                trig=trig,
                coeffs_internal=True,
                apply_scalxc=False,
                s=s,
            )
        )
        odd_p = np.asarray(
            vmec_realspace_synthesis_dzeta_phys(
                coeff_cos=zeros,
                coeff_sin=coeff_sin_stack,
                modes=geom_modes,
                trig=trig,
                coeffs_internal=True,
                apply_scalxc=False,
                s=s,
            )
        )
    else:
        # Match VMEC/bcovar conventions:
        # - even components use physical coefficients (no scalxc),
        # - odd components use internal coefficients (apply scalxc).
        even_base = np.asarray(_eval_mask(mask_even_f, deriv="base", apply_scalxc_local=False))
        even_t = np.asarray(_eval_mask(mask_even_f, deriv="dtheta", apply_scalxc_local=False))
        even_p = np.asarray(_eval_mask(mask_even_f, deriv="dzeta", apply_scalxc_local=False))

        odd_m1_base = np.asarray(_eval_mask(mask_m1_f, deriv="base", apply_scalxc_local=bool(apply_scalxc)))
        odd_m1_t = np.asarray(_eval_mask(mask_m1_f, deriv="dtheta", apply_scalxc_local=bool(apply_scalxc)))
        odd_m1_p = np.asarray(_eval_mask(mask_m1_f, deriv="dzeta", apply_scalxc_local=bool(apply_scalxc)))

        odd_rest_base = np.asarray(_eval_mask(mask_odd_rest_f, deriv="base", apply_scalxc_local=bool(apply_scalxc)))
        odd_rest_t = np.asarray(_eval_mask(mask_odd_rest_f, deriv="dtheta", apply_scalxc_local=bool(apply_scalxc)))
        odd_rest_p = np.asarray(_eval_mask(mask_odd_rest_f, deriv="dzeta", apply_scalxc_local=bool(apply_scalxc)))

        odd_base = odd_m1_base + odd_rest_base
        odd_t = odd_m1_t + odd_rest_t
        odd_p = odd_m1_p + odd_rest_p
        if odd_base.shape[0] >= 2:
            # VMEC axis convention: copy m=1 odd field from js=2 to js=1.
            # Axis corresponds to the radial index (js=1 -> index 0).
            odd_base[0] = odd_m1_base[1]
            odd_t[0] = odd_m1_t[1]
            odd_p[0] = odd_m1_p[1]

    R_even = even_base[0]
    Z_even = even_base[1]
    Ru_even = even_t[0]
    Zu_even = even_t[1]
    Rv_even = even_p[0]
    Zv_even = even_p[1]

    R1 = odd_base[0]
    Z1 = odd_base[1]
    Ru1 = odd_t[0]
    Zu1 = odd_t[1]
    Rv1 = odd_p[0]
    Zv1 = odd_p[1]

    s = np.asarray(s, dtype=float)
    if s.shape[0] < 2:
        return np.zeros_like(np.asarray(bsupu, dtype=float))

    # VMEC's bss.f uses internal even/odd components with explicit shalf factors.
    # See jacobian.f: rs/zs use shalf scaling; rs12/zs12 include d(shalf)/ds.
    hs = float(s[1] - s[0])
    ohs = 1.0 / hs
    dphids = 0.25
    s_half = 0.5 * (s[1:] + s[:-1])
    shalf = np.zeros_like(s, dtype=float)
    shalf[1:] = np.sqrt(np.maximum(s_half, 0.0))
    sh = shalf[:, None, None]

    rv12 = np.zeros_like(R_even, dtype=float)
    zv12 = np.zeros_like(Z_even, dtype=float)
    rs12 = np.zeros_like(R_even, dtype=float)
    zs12 = np.zeros_like(Z_even, dtype=float)

    use_parity_geom_full = (
        isinstance(geom, dict)
        and ("pr1_even" in geom)
        and ("pr1_odd" in geom)
        and ("pz1_even" in geom)
        and ("pz1_odd" in geom)
        and ("pru_even" in geom)
        and ("pru_odd" in geom)
        and ("pzu_even" in geom)
        and ("pzu_odd" in geom)
    )
    # Prefer parity-geometry inputs for bss when available. VMEC's bss.f
    # uses the realspace (symforce) parity fields directly, so default to
    # that behavior unless explicitly disabled.
    use_parity_bss = use_parity_geom_full and (os.getenv("VMEC_JAX_BSS_FROM_PARITY_GEOM", "1") not in ("", "0"))
    use_force_terms = (
        force_rs is not None and force_zs is not None and force_ru12 is not None and force_zu12 is not None
    )
    if use_force_terms:
        use_parity_bss = False

    # Use force-kernel R/Z arrays (VMEC bss.f path) when supplied.
    if use_parity_bss:
        pr1_even = np.asarray(geom["pr1_even"], dtype=float)
        pr1_odd = np.asarray(geom["pr1_odd"], dtype=float)
        pz1_even = np.asarray(geom["pz1_even"], dtype=float)
        pz1_odd = np.asarray(geom["pz1_odd"], dtype=float)
        pru_even = np.asarray(geom["pru_even"], dtype=float)
        pru_odd = np.asarray(geom["pru_odd"], dtype=float)
        pzu_even = np.asarray(geom["pzu_even"], dtype=float)
        pzu_odd = np.asarray(geom["pzu_odd"], dtype=float)

        # The parity fields are built from internal coefficients with VMEC's
        # scalxc applied (odd-m scaled by 1/max(sqrt(s), sqrt(s2))). bss.f
        # expects the *internal* odd fields (before scalxc), so undo it here
        # when the compatibility flag is enabled.
        pr1_odd, pz1_odd, pru_odd, pzu_odd = _undo_bss_scalxc_if_enabled(
            s,
            pr1_odd,
            pz1_odd,
            pru_odd,
            pzu_odd,
        )

        ru12 = np.zeros_like(R_even, dtype=float)
        zu12 = np.zeros_like(Z_even, dtype=float)
        rs = np.zeros_like(R_even, dtype=float)
        zs = np.zeros_like(Z_even, dtype=float)
        ru12[1:] = 0.5 * (pru_even[1:] + pru_even[:-1] + sh[1:] * (pru_odd[1:] + pru_odd[:-1]))
        zu12[1:] = 0.5 * (pzu_even[1:] + pzu_even[:-1] + sh[1:] * (pzu_odd[1:] + pzu_odd[:-1]))
        rs[1:] = ohs * (pr1_even[1:] - pr1_even[:-1] + sh[1:] * (pr1_odd[1:] - pr1_odd[:-1]))
        zs[1:] = ohs * (pz1_even[1:] - pz1_even[:-1] + sh[1:] * (pz1_odd[1:] - pz1_odd[:-1]))
    elif use_force_terms:
        ru12 = np.array(force_ru12, dtype=float, copy=True)
        zu12 = np.array(force_zu12, dtype=float, copy=True)
        rs = np.array(force_rs, dtype=float, copy=True)
        zs = np.array(force_zs, dtype=float, copy=True)
    # Otherwise use half-mesh Jacobian from bcovar when provided to stay
    # consistent with bsupu/bsupv (computed from the same bcovar pipeline).
    elif jac_half is not None:
        ru12 = np.array(jac_half.ru12, dtype=float, copy=True)
        zu12 = np.array(jac_half.zu12, dtype=float, copy=True)
        rs = np.array(jac_half.rs, dtype=float, copy=True)
        zs = np.array(jac_half.zs, dtype=float, copy=True)
    else:
        ru12 = np.zeros_like(R_even, dtype=float)
        zu12 = np.zeros_like(Z_even, dtype=float)
        rs = np.zeros_like(R_even, dtype=float)
        zs = np.zeros_like(Z_even, dtype=float)
        ru12[1:] = 0.5 * (Ru_even[1:] + Ru_even[:-1] + sh[1:] * (Ru1[1:] + Ru1[:-1]))
        zu12[1:] = 0.5 * (Zu_even[1:] + Zu_even[:-1] + sh[1:] * (Zu1[1:] + Zu1[:-1]))
        rs[1:] = ohs * (R_even[1:] - R_even[:-1] + sh[1:] * (R1[1:] - R1[:-1]))
        zs[1:] = ohs * (Z_even[1:] - Z_even[:-1] + sh[1:] * (Z1[1:] - Z1[:-1]))

    use_parity_geom = (
        isinstance(geom, dict)
        and ("pr1_odd" in geom)
        and ("pz1_odd" in geom)
        and ("prv_even" in geom)
        and ("prv_odd" in geom)
        and ("pzv_even" in geom)
        and ("pzv_odd" in geom)
    )
    if use_parity_geom:
        pr1_odd = np.asarray(geom["pr1_odd"], dtype=float)
        pz1_odd = np.asarray(geom["pz1_odd"], dtype=float)
        prv_even = np.asarray(geom["prv_even"], dtype=float)
        prv_odd = np.asarray(geom["prv_odd"], dtype=float)
        pzv_even = np.asarray(geom["pzv_even"], dtype=float)
        pzv_odd = np.asarray(geom["pzv_odd"], dtype=float)

        pr1_odd, pz1_odd, prv_odd, pzv_odd = _undo_bss_scalxc_if_enabled(
            s,
            pr1_odd,
            pz1_odd,
            prv_odd,
            pzv_odd,
        )

        rv12[1:] = 0.5 * (prv_even[1:] + prv_even[:-1] + sh[1:] * (prv_odd[1:] + prv_odd[:-1]))
        zv12[1:] = 0.5 * (pzv_even[1:] + pzv_even[:-1] + sh[1:] * (pzv_odd[1:] + pzv_odd[:-1]))
        rs12[1:] = rs[1:] + dphids * (pr1_odd[1:] + pr1_odd[:-1]) / sh[1:]
        zs12[1:] = zs[1:] + dphids * (pz1_odd[1:] + pz1_odd[:-1]) / sh[1:]
    else:
        rv12[1:] = 0.5 * (Rv_even[1:] + Rv_even[:-1] + sh[1:] * (Rv1[1:] + Rv1[:-1]))
        zv12[1:] = 0.5 * (Zv_even[1:] + Zv_even[:-1] + sh[1:] * (Zv1[1:] + Zv1[:-1]))

        rs12[1:] = rs[1:] + dphids * (R1[1:] + R1[:-1]) / sh[1:]
        zs12[1:] = zs[1:] + dphids * (Z1[1:] + Z1[:-1]) / sh[1:]

    # Axis fill: mirror js=2 into js=1 (VMEC convention).
    if rs12.shape[0] > 1:
        rs12[0] = rs12[1]
        zs12[0] = zs12[1]
        ru12[0] = ru12[1]
        zu12[0] = zu12[1]
        rv12[0] = rv12[1]
        zv12[0] = zv12[1]

    g_su = rs12 * ru12 + zs12 * zu12
    g_sv = rs12 * rv12 + zs12 * zv12
    bsubs = np.asarray(bsupu, dtype=float) * g_su + np.asarray(bsupv, dtype=float) * g_sv

    if os.getenv("VMEC_JAX_DUMP_BSS_INPUTS", "") not in ("", "0"):
        outdir = Path(os.getenv("VMEC_JAX_DUMP_DIR", ".")).expanduser().resolve()
        outdir.mkdir(parents=True, exist_ok=True)
        tag = os.getenv("VMEC_JAX_DUMP_TAG", "").strip()
        name = "bss_inputs_jax" + (f"_{tag}" if tag else "") + ".dat"
        path = outdir / name
        r12 = None
        if jac_half is not None:
            try:
                r12 = np.asarray(jac_half.r12, dtype=float)
            except Exception:
                r12 = None
        if r12 is None:
            r12 = np.zeros_like(R_even, dtype=float)
            r12[1:] = 0.5 * (R_even[1:] + R_even[:-1] + sh[1:] * (R1[1:] + R1[:-1]))
            r12[0] = r12[1]
        with path.open("w") as f:
            f.write("# bss inputs dump (half mesh)\n")
            f.write(f"ns={r12.shape[0]}\n")
            f.write(f"ntheta3={r12.shape[1]}\n")
            f.write(f"nzeta={r12.shape[2]}\n")
            f.write("columns: js lt lz r12 rs zs ru12 zu12 bsupu bsupv\n")
            ns, ntheta3, nzeta = r12.shape
            for lt in range(ntheta3):
                for lz in range(nzeta):
                    for js in range(ns):
                        bsupu_val = float(np.asarray(bsupu, dtype=float)[js, lt, lz])
                        bsupv_val = float(np.asarray(bsupv, dtype=float)[js, lt, lz])
                        f.write(
                            f"{js + 1:6d}{lt + 1:6d}{lz + 1:6d}"
                            f"{r12[js, lt, lz]:24.16E}{rs[js, lt, lz]:24.16E}{zs[js, lt, lz]:24.16E}"
                            f"{ru12[js, lt, lz]:24.16E}{zu12[js, lt, lz]:24.16E}"
                            f"{bsupu_val:24.16E}{bsupv_val:24.16E}\n"
                        )

    if os.getenv("VMEC_JAX_DUMP_BSS_TERMS", "") not in ("", "0"):
        outdir = Path(os.getenv("VMEC_JAX_DUMP_DIR", ".")).expanduser().resolve()
        outdir.mkdir(parents=True, exist_ok=True)
        tag = os.getenv("VMEC_JAX_DUMP_TAG", "").strip()
        name = "bss_terms_jax" + (f"_{tag}" if tag else "") + ".npz"
        pr1_even = (
            np.asarray(geom.get("pr1_even"), dtype=float)
            if isinstance(geom, dict) and "pr1_even" in geom
            else np.zeros_like(R_even, dtype=float)
        )
        pr1_odd = (
            np.asarray(geom.get("pr1_odd"), dtype=float)
            if isinstance(geom, dict) and "pr1_odd" in geom
            else np.zeros_like(R_even, dtype=float)
        )
        prv_even = (
            np.asarray(geom.get("prv_even"), dtype=float)
            if isinstance(geom, dict) and "prv_even" in geom
            else np.zeros_like(R_even, dtype=float)
        )
        prv_odd = (
            np.asarray(geom.get("prv_odd"), dtype=float)
            if isinstance(geom, dict) and "prv_odd" in geom
            else np.zeros_like(R_even, dtype=float)
        )
        pru_even = (
            np.asarray(geom.get("pru_even"), dtype=float)
            if isinstance(geom, dict) and "pru_even" in geom
            else np.zeros_like(R_even, dtype=float)
        )
        pru_odd = (
            np.asarray(geom.get("pru_odd"), dtype=float)
            if isinstance(geom, dict) and "pru_odd" in geom
            else np.zeros_like(R_even, dtype=float)
        )
        pzu_even = (
            np.asarray(geom.get("pzu_even"), dtype=float)
            if isinstance(geom, dict) and "pzu_even" in geom
            else np.zeros_like(R_even, dtype=float)
        )
        pzu_odd = (
            np.asarray(geom.get("pzu_odd"), dtype=float)
            if isinstance(geom, dict) and "pzu_odd" in geom
            else np.zeros_like(R_even, dtype=float)
        )
        np.savez(
            outdir / name,
            r1_even=np.asarray(R_even, dtype=float),
            r1_odd=np.asarray(R1, dtype=float),
            pr1_even=pr1_even,
            pr1_odd=pr1_odd,
            rv_even=np.asarray(Rv_even, dtype=float),
            rv_odd=np.asarray(Rv1, dtype=float),
            prv_even=prv_even,
            prv_odd=prv_odd,
            pru_even=pru_even,
            pru_odd=pru_odd,
            pzu_even=pzu_even,
            pzu_odd=pzu_odd,
            rs12=np.asarray(rs12, dtype=float),
            zs12=np.asarray(zs12, dtype=float),
            rv12=np.asarray(rv12, dtype=float),
            zv12=np.asarray(zv12, dtype=float),
            ru12=np.asarray(ru12, dtype=float),
            zu12=np.asarray(zu12, dtype=float),
            gsu=np.asarray(g_su, dtype=float),
            gsv=np.asarray(g_sv, dtype=float),
            bsubs=np.asarray(bsubs, dtype=float),
            bsupu=np.asarray(bsupu, dtype=float),
            bsupv=np.asarray(bsupv, dtype=float),
            s=np.asarray(s, dtype=float),
        )

    return bsubs


def _bsubuv_parity_from_state(
    *,
    state: VMECState,
    geom_modes,
    trig,
    s: np.ndarray,
    lconm1: bool,
    lthreed: bool,
    lasym: bool,
    bsupu: np.ndarray,
    bsupv: np.ndarray,
    lu1_full: np.ndarray,
    lv1_full: np.ndarray,
    sqrtg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct parity-separated bsubu/bsubv using VMEC internal even/odd splitting."""
    from .vmec_realspace import (
        vmec_realspace_synthesis,
        vmec_realspace_synthesis_dtheta,
    )
    from .vmec_jacobian import _apply_vmec_axis_rules

    m = np.asarray(geom_modes.m, dtype=int)
    mask_even = (m % 2) == 0
    mask_odd = ~mask_even

    Rcos = np.asarray(state.Rcos)
    Rsin = np.asarray(state.Rsin)
    Zcos = np.asarray(state.Zcos)
    Zsin = np.asarray(state.Zsin)
    if bool(lconm1):
        Rcos, Zsin, Rsin, Zcos = vmec_m1_internal_to_physical_signed(
            Rcos=Rcos,
            Zsin=Zsin,
            Rsin=Rsin,
            Zcos=Zcos,
            modes=geom_modes,
            lthreed=bool(lthreed),
            lasym=bool(lasym),
            lconm1=bool(lconm1),
        )

    Rcos = _apply_vmec_axis_rules(Rcos, m)
    Rsin = _apply_vmec_axis_rules(Rsin, m)
    Zcos = _apply_vmec_axis_rules(Zcos, m)
    Zsin = _apply_vmec_axis_rules(Zsin, m)

    coeff_cos_stack = np.stack([Rcos, Zcos], axis=0)
    coeff_sin_stack = np.stack([Rsin, Zsin], axis=0)
    mask_stack = np.stack([mask_even.astype(float), mask_odd.astype(float)], axis=0)
    coeff_cos = coeff_cos_stack[None, ...] * mask_stack[:, None, None, :]
    coeff_sin = coeff_sin_stack[None, ...] * mask_stack[:, None, None, :]

    stack = vmec_realspace_synthesis(
        coeff_cos=coeff_cos,
        coeff_sin=coeff_sin,
        modes=geom_modes,
        trig=trig,
        coeffs_internal=True,
        apply_scalxc=True,
        s=s,
    )
    stack_t = vmec_realspace_synthesis_dtheta(
        coeff_cos=coeff_cos,
        coeff_sin=coeff_sin,
        modes=geom_modes,
        trig=trig,
        coeffs_internal=True,
        apply_scalxc=True,
        s=s,
    )
    stack_p = vmec_realspace_synthesis_dzeta_phys(
        coeff_cos=coeff_cos,
        coeff_sin=coeff_sin,
        modes=geom_modes,
        trig=trig,
        coeffs_internal=True,
        apply_scalxc=True,
        s=s,
    )

    even = np.asarray(stack[0])
    odd = np.asarray(stack[1])
    even_t = np.asarray(stack_t[0])
    odd_t = np.asarray(stack_t[1])
    even_p = np.asarray(stack_p[0])
    odd_p = np.asarray(stack_p[1])

    Ru_even = even_t[0]
    Ru_odd = odd_t[0]
    Zu_even = even_t[1]
    Zu_odd = odd_t[1]
    Rv_even = even_p[0]
    Rv_odd = odd_p[0]
    Zv_even = even_p[1]
    Zv_odd = odd_p[1]

    pshalf = _pshalf_from_s(np.asarray(s, dtype=float))[:, None, None]
    # bsubu/bsubv live on the radial half mesh in VMEC. Their parity algebra
    # must therefore use s_half = pshalf^2 (not full-mesh s_j).
    s_term = pshalf * pshalf

    guu_even = Ru_even * Ru_even + Zu_even * Zu_even + s_term * (Ru_odd * Ru_odd + Zu_odd * Zu_odd)
    guu_odd = 2.0 * (Ru_even * Ru_odd + Zu_even * Zu_odd)
    guv_even = Ru_even * Rv_even + Zu_even * Zv_even + s_term * (Ru_odd * Rv_odd + Zu_odd * Zv_odd)
    guv_odd = Ru_even * Rv_odd + Ru_odd * Rv_even + Zu_even * Zv_odd + Zu_odd * Zv_even
    gvv_even = Rv_even * Rv_even + Zv_even * Zv_even + s_term * (Rv_odd * Rv_odd + Zv_odd * Zv_odd)
    gvv_odd = 2.0 * (Rv_even * Rv_odd + Zv_even * Zv_odd)

    overg = np.where(np.asarray(sqrtg) != 0.0, 1.0 / np.asarray(sqrtg), 0.0)
    bsupu_even = np.asarray(bsupu, dtype=float)
    bsupv_even = np.asarray(bsupv, dtype=float)
    bsupu_odd = np.zeros_like(bsupu_even)
    bsupv_odd = np.zeros_like(bsupv_even)
    if int(bsupu_even.shape[0]) >= 2:
        avg_lv1 = np.asarray(lv1_full[1:] + lv1_full[:-1], dtype=float)
        avg_lu1 = np.asarray(lu1_full[1:] + lu1_full[:-1], dtype=float)
        bsupu_odd[1:] = 0.5 * overg[1:] * avg_lv1
        bsupv_odd[1:] = 0.5 * overg[1:] * avg_lu1
        bsupu_even = bsupu_even - pshalf * bsupu_odd
        bsupv_even = bsupv_even - pshalf * bsupv_odd

    bsubu_even = (
        guu_even * bsupu_even + s_term * guu_odd * bsupu_odd + guv_even * bsupv_even + s_term * guv_odd * bsupv_odd
    )
    bsubu_odd = guu_even * bsupu_odd + guu_odd * bsupu_even + guv_even * bsupv_odd + guv_odd * bsupv_even
    bsubv_even = (
        guv_even * bsupu_even + s_term * guv_odd * bsupu_odd + gvv_even * bsupv_even + s_term * gvv_odd * bsupv_odd
    )
    bsubv_odd = guv_even * bsupu_odd + guv_odd * bsupu_even + gvv_even * bsupv_odd + gvv_odd * bsupv_even

    return bsubu_even, bsubu_odd, bsubv_even, bsubv_odd


def _bsubuv_parity_from_coeffs(
    *,
    bsubumnc: np.ndarray,
    bsubumns: np.ndarray,
    bsubvmnc: np.ndarray,
    bsubvmns: np.ndarray,
    modes,
    trig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split bsubu/bsubv into even/odd m parity using Fourier coefficients."""
    m = np.asarray(modes.m, dtype=int)
    mask_even = (m % 2) == 0
    mask_odd = ~mask_even
    mask_even = mask_even[None, :]
    mask_odd = mask_odd[None, :]

    bsubumnc = np.asarray(bsubumnc, dtype=float)
    bsubumns = np.asarray(bsubumns, dtype=float)
    bsubvmnc = np.asarray(bsubvmnc, dtype=float)
    bsubvmns = np.asarray(bsubvmns, dtype=float)

    bsubumnc_even = bsubumnc * mask_even
    bsubumns_even = bsubumns * mask_even
    bsubumnc_odd = bsubumnc * mask_odd
    bsubumns_odd = bsubumns * mask_odd

    bsubvmnc_even = bsubvmnc * mask_even
    bsubvmns_even = bsubvmns * mask_even
    bsubvmnc_odd = bsubvmnc * mask_odd
    bsubvmns_odd = bsubvmns * mask_odd

    # Use wrout-style Nyquist synthesis instead of generic helical eval so the
    # parity split stays on VMEC's reduced-grid normalization.
    bsubu_even = _vmec_wrout_nyquist_synthesis(
        coeff_c=bsubumnc_even,
        coeff_s=bsubumns_even,
        modes=modes,
        trig=trig,
    )
    bsubu_odd = _vmec_wrout_nyquist_synthesis(
        coeff_c=bsubumnc_odd,
        coeff_s=bsubumns_odd,
        modes=modes,
        trig=trig,
    )
    bsubv_even = _vmec_wrout_nyquist_synthesis(
        coeff_c=bsubvmnc_even,
        coeff_s=bsubvmns_even,
        modes=modes,
        trig=trig,
    )
    bsubv_odd = _vmec_wrout_nyquist_synthesis(
        coeff_c=bsubvmnc_odd,
        coeff_s=bsubvmns_odd,
        modes=modes,
        trig=trig,
    )
    return bsubu_even, bsubu_odd, bsubv_even, bsubv_odd


def _bsubuv_parity_from_realspace_jxbforce(
    *,
    bsubu: np.ndarray,
    bsubv: np.ndarray,
    trig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Recover jxbforce parity channels directly from real-space bsubu/bsubv."""
    # jdotb can be cancellation-limited near the edge. Keep this projection in
    # long-double to reduce parity-channel roundoff before the jxbforce filter.
    acc_dtype = np.longdouble
    bsubu = np.asarray(bsubu, dtype=acc_dtype)
    bsubv = np.asarray(bsubv, dtype=acc_dtype)
    if bsubu.shape != bsubv.shape:
        raise ValueError("bsubu/bsubv shape mismatch")
    if bsubu.ndim != 3:
        raise ValueError("Expected bsubu/bsubv with shape (ns, ntheta, nzeta)")

    ns, ntheta, nzeta = bsubu.shape
    nt2 = int(trig.ntheta2)
    if ntheta < nt2:
        raise ValueError("bsubu grid smaller than ntheta2")

    mnyq, nnyq = _jxbforce_nyquist_limits(trig)
    mmax = int(max(mnyq, 0))
    nmax = int(max(nnyq, 0))

    cosmui = np.asarray(trig.cosmui, dtype=acc_dtype)[:nt2, : mmax + 1]
    sinmui = np.asarray(trig.sinmui, dtype=acc_dtype)[:nt2, : mmax + 1]
    cosmu = np.asarray(trig.cosmu, dtype=acc_dtype)[:nt2, : mmax + 1]
    sinmu = np.asarray(trig.sinmu, dtype=acc_dtype)[:nt2, : mmax + 1]
    cosnv = np.asarray(trig.cosnv, dtype=acc_dtype)[:, : nmax + 1]
    sinnv = np.asarray(trig.sinnv, dtype=acc_dtype)[:, : nmax + 1]

    r0scale = float(getattr(trig, "r0scale", 1.0))
    base_dnorm = acc_dtype(1.0) / acc_dtype(r0scale**2)

    bsubu_even = np.zeros((ns, nt2, nzeta), dtype=acc_dtype)
    bsubu_odd = np.zeros((ns, nt2, nzeta), dtype=acc_dtype)
    bsubv_even = np.zeros((ns, nt2, nzeta), dtype=acc_dtype)
    bsubv_odd = np.zeros((ns, nt2, nzeta), dtype=acc_dtype)

    for js in range(ns):
        bu = bsubu[js, :nt2, :]
        bv = bsubv[js, :nt2, :]
        for m in range(mmax + 1):
            use_odd = (m % 2) == 1
            for n in range(nmax + 1):
                dnorm1 = base_dnorm
                if mnyq > 0 and m == mnyq:
                    dnorm1 *= 0.5
                if nnyq > 0 and n == nnyq and n != 0:
                    dnorm1 *= 0.5

                bsubumn1 = acc_dtype(0.0)
                bsubumn2 = acc_dtype(0.0)
                bsubvmn1 = acc_dtype(0.0)
                bsubvmn2 = acc_dtype(0.0)
                for k in range(nzeta):
                    for j in range(nt2):
                        tcosi1 = cosmui[j, m] * cosnv[k, n] * dnorm1
                        tcosi2 = sinmui[j, m] * sinnv[k, n] * dnorm1
                        val_u = bu[j, k]
                        val_v = bv[j, k]
                        bsubumn1 += tcosi1 * val_u
                        bsubumn2 += tcosi2 * val_u
                        bsubvmn1 += tcosi1 * val_v
                        bsubvmn2 += tcosi2 * val_v

                for k in range(nzeta):
                    for j in range(nt2):
                        tcos1 = cosmu[j, m] * cosnv[k, n]
                        tcos2 = sinmu[j, m] * sinnv[k, n]
                        ucontrib = tcos1 * bsubumn1 + tcos2 * bsubumn2
                        vcontrib = tcos1 * bsubvmn1 + tcos2 * bsubvmn2
                        if use_odd:
                            bsubu_odd[js, j, k] += ucontrib
                            bsubv_odd[js, j, k] += vcontrib
                        else:
                            bsubu_even[js, j, k] += ucontrib
                            bsubv_even[js, j, k] += vcontrib

    return (
        np.asarray(bsubu_even, dtype=float),
        np.asarray(bsubu_odd, dtype=float),
        np.asarray(bsubv_even, dtype=float),
        np.asarray(bsubv_odd, dtype=float),
    )


def _bsubuv_parity_from_bcovar(
    *,
    bsubu_even: np.ndarray,
    bsubv_even: np.ndarray,
    s: np.ndarray,
    iequi: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct parity-separated bsubu/bsubv from bcovar even components."""
    s_full = np.asarray(s, dtype=float)
    psqrts = np.sqrt(np.maximum(s_full, 0.0))[:, None, None]
    pshalf = _pshalf_from_s(s_full)[:, None, None]
    scale = pshalf if int(iequi) == 1 else psqrts
    bsubu_even = np.asarray(bsubu_even, dtype=float)
    bsubv_even = np.asarray(bsubv_even, dtype=float)
    bsubu_odd = scale * bsubu_even
    bsubv_odd = scale * bsubv_even
    return bsubu_even, bsubu_odd, bsubv_even, bsubv_odd


def _filter_bsubuv_jxbforce(
    *,
    bsubu: np.ndarray,
    bsubv: np.ndarray,
    trig,
    nfp: int,
    mmax_force: int,
    nmax_force: int,
    s: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """JXBFORCE-style low-pass filter for bsubu/bsubv (lasym=False)."""
    # For parity-critical diagnostics we prefer the explicit VMEC loop
    # ordering, which matches the Fortran summation order more closely.
    import os

    # Default to the vectorized path for performance; the loop-based path is
    # retained for parity debugging.
    use_loop = os.getenv("VMEC_JAX_BSUB_FILTER_LOOP", "0") not in ("", "0")
    if use_loop:
        return _filter_bsubuv_jxbforce_loop(
            bsubu=bsubu,
            bsubv=bsubv,
            trig=trig,
            mmax_force=mmax_force,
            nmax_force=nmax_force,
            s=s,
        )
    bsubu = np.asarray(bsubu, dtype=float)
    bsubv = np.asarray(bsubv, dtype=float)
    ns, ntheta, nzeta = bsubu.shape

    nt2 = int(trig.ntheta2)
    if ntheta < nt2:
        raise ValueError("bsubu grid smaller than ntheta2")

    mmax = int(mmax_force)
    nmax = int(nmax_force)
    if mmax < 0 or nmax < 0:
        return bsubu[:, :nt2, :].copy(), bsubv[:, :nt2, :].copy()

    # Implement the jxbforce low-pass filter explicitly to match VMEC's
    # parity-normalized Fourier transforms.
    bsubu_red = np.asarray(bsubu[:, :nt2, :], dtype=float)
    bsubv_red = np.asarray(bsubv[:, :nt2, :], dtype=float)

    cosmui = np.asarray(trig.cosmui, dtype=float)[:nt2, : mmax + 1]
    sinmui = np.asarray(trig.sinmui, dtype=float)[:nt2, : mmax + 1]
    cosmu = np.asarray(trig.cosmu, dtype=float)[:nt2, : mmax + 1]
    sinmu = np.asarray(trig.sinmu, dtype=float)[:nt2, : mmax + 1]
    cosnv = np.asarray(trig.cosnv, dtype=float)[:, : nmax + 1]
    sinnv = np.asarray(trig.sinnv, dtype=float)[:, : nmax + 1]

    r0scale = float(getattr(trig, "r0scale", 1.0))
    dnorm1 = 1.0 / (r0scale**2)
    dmult = np.full((mmax + 1, nmax + 1), dnorm1, dtype=float)
    mnyq, nnyq = _jxbforce_nyquist_limits(trig)
    if mnyq > 0 and mnyq <= mmax:
        dmult[mnyq, :] *= 0.5
    if nnyq > 0 and nnyq <= nmax:
        dmult[:, nnyq] *= 0.5

    def _pshalf_from_s(s_full: np.ndarray) -> np.ndarray:
        if s_full.shape[0] < 2:
            return np.sqrt(np.maximum(s_full, 0.0))
        sh = 0.5 * (s_full[1:] + s_full[:-1])
        p = np.concatenate([sh[:1], sh], axis=0)
        return np.sqrt(np.maximum(p, 0.0))

    pshalf = None
    if s is not None:
        s_full = np.asarray(s, dtype=float)
        pshalf = _pshalf_from_s(s_full)
        # Avoid divide-by-zero on-axis; VMEC sets shalf(1)=shalf(2).
        if pshalf.shape[0] > 1:
            pshalf[0] = pshalf[1]

    # When the filter limits match the available basis, the transform should
    # be identity (avoid numerical drift by returning the original fields).
    full_mmax, full_nmax = _jxbforce_nyquist_limits(trig)
    if mmax >= full_mmax and nmax >= full_nmax:
        return bsubu[:, :nt2, :].copy(), bsubv[:, :nt2, :].copy()

    def _filter_one(f: np.ndarray) -> np.ndarray:
        # Forward transform: cos(mu)cos(nv) + sin(mu)sin(nv) (jxbforce).
        f_theta_cos = np.einsum("sik,im->smk", f, cosmui, optimize=True)
        f_theta_sin = np.einsum("sik,im->smk", f, sinmui, optimize=True)
        coeff1 = np.einsum("smk,kn->smn", f_theta_cos, cosnv, optimize=True)
        coeff2 = np.einsum("smk,kn->smn", f_theta_sin, sinnv, optimize=True)
        coeff1 = coeff1 * dmult[None, :, :]
        coeff2 = coeff2 * dmult[None, :, :]
        # VMEC stores odd-m fields with an extra sqrt(s) factor. Undo that
        # scaling for odd m before the inverse transform (jxbforce does this
        # via bsubu(js,:,1)/shalf(js)).
        if pshalf is not None and mmax >= 1:
            odd = (np.arange(mmax + 1) % 2) == 1
            if np.any(odd):
                scale = np.ones((coeff1.shape[0], mmax + 1, 1), dtype=float)
                scale[:, odd, 0] = 1.0 / pshalf[:, None]
                coeff1 = coeff1 * scale
                coeff2 = coeff2 * scale

        # Inverse transform back to real space on the reduced grid.
        tmp_cos = np.einsum("smn,im->sin", coeff1, cosmu, optimize=True)
        tmp_sin = np.einsum("smn,im->sin", coeff2, sinmu, optimize=True)
        return np.einsum("sin,kn->sik", tmp_cos, cosnv, optimize=True) + np.einsum(
            "sin,kn->sik", tmp_sin, sinnv, optimize=True
        )

    return _filter_one(bsubu_red), _filter_one(bsubv_red)


def _filter_bsubuv_jxbforce_parity(
    *,
    bsubu_even: np.ndarray,
    bsubu_odd: np.ndarray,
    bsubv_even: np.ndarray,
    bsubv_odd: np.ndarray,
    trig,
    mmax_force: int,
    nmax_force: int,
    s: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """JXBFORCE-style low-pass filter using parity-separated bsubu/bsubv (lasym=False).

    This is a vectorized equivalent of :func:`_filter_bsubuv_jxbforce_parity_loop`.
    We keep the loop-based implementation for parity debugging, but default to the
    vectorized path for performance.
    """
    import os

    use_loop = os.getenv("VMEC_JAX_BSUB_FILTER_LOOP", "0") not in ("", "0")
    if use_loop:
        return _filter_bsubuv_jxbforce_parity_loop(
            bsubu_even=bsubu_even,
            bsubu_odd=bsubu_odd,
            bsubv_even=bsubv_even,
            bsubv_odd=bsubv_odd,
            trig=trig,
            mmax_force=mmax_force,
            nmax_force=nmax_force,
            s=s,
        )

    bsubu_even = np.asarray(bsubu_even, dtype=float)
    bsubu_odd = np.asarray(bsubu_odd, dtype=float)
    bsubv_even = np.asarray(bsubv_even, dtype=float)
    bsubv_odd = np.asarray(bsubv_odd, dtype=float)
    if bsubu_even.shape != bsubu_odd.shape or bsubu_even.shape != bsubv_even.shape:
        raise ValueError("Parity bsubu/bsubv shape mismatch")

    ns, ntheta, nzeta = bsubu_even.shape
    nt2 = int(trig.ntheta2)
    if ntheta < nt2:
        raise ValueError("bsubu grid smaller than ntheta2")

    mmax = int(mmax_force)
    nmax = int(nmax_force)
    if mmax < 0 or nmax < 0:
        return bsubu_even[:, :nt2, :].copy(), bsubv_even[:, :nt2, :].copy()

    bsubu_even_red = bsubu_even[:, :nt2, :]
    bsubu_odd_red = bsubu_odd[:, :nt2, :]
    bsubv_even_red = bsubv_even[:, :nt2, :]
    bsubv_odd_red = bsubv_odd[:, :nt2, :]

    cosmui = np.asarray(trig.cosmui, dtype=float)[:nt2, : mmax + 1]
    sinmui = np.asarray(trig.sinmui, dtype=float)[:nt2, : mmax + 1]
    cosmu = np.asarray(trig.cosmu, dtype=float)[:nt2, : mmax + 1]
    sinmu = np.asarray(trig.sinmu, dtype=float)[:nt2, : mmax + 1]
    cosnv = np.asarray(trig.cosnv, dtype=float)[:, : nmax + 1]
    sinnv = np.asarray(trig.sinnv, dtype=float)[:, : nmax + 1]

    r0scale = float(getattr(trig, "r0scale", 1.0))
    dnorm1 = 1.0 / (r0scale**2)
    dmult = np.full((mmax + 1, nmax + 1), dnorm1, dtype=float)
    mnyq, nnyq = _jxbforce_nyquist_limits(trig)
    if mnyq > 0 and mnyq <= mmax:
        dmult[mnyq, :] *= 0.5
    if nnyq > 0 and nnyq <= nmax:
        dmult[:, nnyq] *= 0.5

    pshalf = None
    if s is not None:
        s_full = np.asarray(s, dtype=float)
        pshalf = _pshalf_from_s(s_full)
        if pshalf.shape[0] > 1:
            pshalf[0] = pshalf[1]

    odd_m = (np.arange(mmax + 1) % 2) == 1

    def _forward(f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        f_theta_cos = np.einsum("sik,im->smk", f, cosmui, optimize=True)
        f_theta_sin = np.einsum("sik,im->smk", f, sinmui, optimize=True)
        coeff1 = np.einsum("smk,kn->smn", f_theta_cos, cosnv, optimize=True)
        coeff2 = np.einsum("smk,kn->smn", f_theta_sin, sinnv, optimize=True)
        coeff1 = coeff1 * dmult[None, :, :]
        coeff2 = coeff2 * dmult[None, :, :]
        return coeff1, coeff2

    def _inverse(coeff1: np.ndarray, coeff2: np.ndarray) -> np.ndarray:
        tmp_cos = np.einsum("smn,im->sin", coeff1, cosmu, optimize=True)
        tmp_sin = np.einsum("smn,im->sin", coeff2, sinmu, optimize=True)
        return np.einsum("sin,kn->sik", tmp_cos, cosnv, optimize=True) + np.einsum(
            "sin,kn->sik", tmp_sin, sinnv, optimize=True
        )

    def _filter_field(f_even: np.ndarray, f_odd: np.ndarray) -> np.ndarray:
        c1e, c2e = _forward(f_even)
        c1o, c2o = _forward(f_odd)
        if pshalf is not None and mmax >= 1 and np.any(odd_m):
            scale = pshalf[:, None, None]
            c1o[:, odd_m, :] = c1o[:, odd_m, :] / scale
            c2o[:, odd_m, :] = c2o[:, odd_m, :] / scale
        if np.any(odd_m):
            c1e = c1e.copy()
            c2e = c2e.copy()
            c1e[:, odd_m, :] = c1o[:, odd_m, :]
            c2e[:, odd_m, :] = c2o[:, odd_m, :]
        return _inverse(c1e, c2e)

    bsubu_out = _filter_field(bsubu_even_red, bsubu_odd_red)
    bsubv_out = _filter_field(bsubv_even_red, bsubv_odd_red)
    return np.asarray(bsubu_out, dtype=float), np.asarray(bsubv_out, dtype=float)


def _filter_bsubuv_jxbforce_parity_loop(
    *,
    bsubu_even: np.ndarray,
    bsubu_odd: np.ndarray,
    bsubv_even: np.ndarray,
    bsubv_odd: np.ndarray,
    trig,
    mmax_force: int,
    nmax_force: int,
    s: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Loop-based jxbforce low-pass filter using parity-separated bsubu/bsubv."""
    # Cancellation in the low-pass Fourier sums can be severe near the edge for
    # high-shear equilibria. Accumulate in long double, cast back to float.
    acc_dtype = np.longdouble
    bsubu_even = np.asarray(bsubu_even, dtype=acc_dtype)
    bsubu_odd = np.asarray(bsubu_odd, dtype=acc_dtype)
    bsubv_even = np.asarray(bsubv_even, dtype=acc_dtype)
    bsubv_odd = np.asarray(bsubv_odd, dtype=acc_dtype)
    if bsubu_even.shape != bsubu_odd.shape or bsubu_even.shape != bsubv_even.shape:
        raise ValueError("Parity bsubu/bsubv shape mismatch")

    ns, ntheta, nzeta = bsubu_even.shape
    nt2 = int(trig.ntheta2)
    if ntheta < nt2:
        raise ValueError("bsubu grid smaller than ntheta2")

    mmax = int(mmax_force)
    nmax = int(nmax_force)
    if mmax < 0 or nmax < 0:
        return bsubu_even[:, :nt2, :].copy(), bsubv_even[:, :nt2, :].copy()

    cosmui = np.asarray(trig.cosmui, dtype=acc_dtype)[:nt2, : mmax + 1]
    sinmui = np.asarray(trig.sinmui, dtype=acc_dtype)[:nt2, : mmax + 1]
    cosmu = np.asarray(trig.cosmu, dtype=acc_dtype)[:nt2, : mmax + 1]
    sinmu = np.asarray(trig.sinmu, dtype=acc_dtype)[:nt2, : mmax + 1]
    cosnv = np.asarray(trig.cosnv, dtype=acc_dtype)[:, : nmax + 1]
    sinnv = np.asarray(trig.sinnv, dtype=acc_dtype)[:, : nmax + 1]

    r0scale = float(getattr(trig, "r0scale", 1.0))
    base_dnorm = acc_dtype(1.0) / acc_dtype(r0scale**2)
    mnyq, nnyq = _jxbforce_nyquist_limits(trig)

    def _pshalf_from_s(s_full: np.ndarray) -> np.ndarray:
        if s_full.shape[0] < 2:
            return np.sqrt(np.maximum(s_full, 0.0))
        sh = 0.5 * (s_full[1:] + s_full[:-1])
        p = np.concatenate([sh[:1], sh], axis=0)
        return np.sqrt(np.maximum(p, 0.0))

    pshalf = None
    if s is not None:
        s_full = np.asarray(s, dtype=acc_dtype)
        pshalf = _pshalf_from_s(s_full)
        if pshalf.shape[0] > 1:
            pshalf[0] = pshalf[1]

    bsubu_out = np.zeros((ns, nt2, nzeta), dtype=acc_dtype)
    bsubv_out = np.zeros((ns, nt2, nzeta), dtype=acc_dtype)

    for js in range(ns):
        bsubu_even_s = bsubu_even[js, :nt2, :]
        bsubu_odd_s = bsubu_odd[js, :nt2, :]
        bsubv_even_s = bsubv_even[js, :nt2, :]
        bsubv_odd_s = bsubv_odd[js, :nt2, :]
        for m in range(mmax + 1):
            use_odd = (m % 2) == 1
            bsubu_in = bsubu_odd_s if use_odd else bsubu_even_s
            bsubv_in = bsubv_odd_s if use_odd else bsubv_even_s
            for n in range(nmax + 1):
                dnorm1 = base_dnorm
                if mnyq > 0 and m == mnyq:
                    dnorm1 *= 0.5
                if nnyq > 0 and n == nnyq and n != 0:
                    dnorm1 *= 0.5
                if use_odd and (pshalf is not None):
                    dnorm1 = dnorm1 / float(pshalf[js])

                bsubumn1 = acc_dtype(0.0)
                bsubumn2 = acc_dtype(0.0)
                bsubvmn1 = acc_dtype(0.0)
                bsubvmn2 = acc_dtype(0.0)

                for k in range(nzeta):
                    for j in range(nt2):
                        tsini1 = sinmui[j, m] * cosnv[k, n] * dnorm1
                        tsini2 = cosmui[j, m] * sinnv[k, n] * dnorm1
                        tcosi1 = cosmui[j, m] * cosnv[k, n] * dnorm1
                        tcosi2 = sinmui[j, m] * sinnv[k, n] * dnorm1
                        val_u = bsubu_in[j, k]
                        val_v = bsubv_in[j, k]
                        bsubumn1 += tcosi1 * val_u
                        bsubumn2 += tcosi2 * val_u
                        bsubvmn1 += tcosi1 * val_v
                        bsubvmn2 += tcosi2 * val_v

                for k in range(nzeta):
                    for j in range(nt2):
                        tcos1 = cosmu[j, m] * cosnv[k, n]
                        tcos2 = sinmu[j, m] * sinnv[k, n]
                        bsubu_out[js, j, k] += tcos1 * bsubumn1 + tcos2 * bsubumn2
                        bsubv_out[js, j, k] += tcos1 * bsubvmn1 + tcos2 * bsubvmn2

    return np.asarray(bsubu_out, dtype=float), np.asarray(bsubv_out, dtype=float)


def _jxbforce_filter_with_bsubs_derivs_loop(
    *,
    bsubs: np.ndarray,
    bsubu_even: np.ndarray,
    bsubu_odd: np.ndarray,
    bsubv_even: np.ndarray,
    bsubv_odd: np.ndarray,
    trig,
    mmax_force: int,
    nmax_force: int,
    s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """VMEC jxbforce low-pass + bsubsu/bsubsv in one loop (lasym=False).

    This mirrors the coupled transform block in ``jxbforce.f`` where filtered
    ``bsubu/bsubv`` and derivatives ``bsubsu/bsubsv`` are reconstructed from the
    same Fourier accumulators. Keeping these coupled reduces cancellation drift
    in downstream ``itheta/izeta/bdotk`` diagnostics.
    """
    acc_dtype = np.longdouble
    bsubs = np.asarray(bsubs, dtype=acc_dtype)
    bsubu_even = np.asarray(bsubu_even, dtype=acc_dtype)
    bsubu_odd = np.asarray(bsubu_odd, dtype=acc_dtype)
    bsubv_even = np.asarray(bsubv_even, dtype=acc_dtype)
    bsubv_odd = np.asarray(bsubv_odd, dtype=acc_dtype)

    if bsubu_even.shape != bsubu_odd.shape or bsubu_even.shape != bsubv_even.shape:
        raise ValueError("Parity bsubu/bsubv shape mismatch")
    if bsubs.shape[:2] != bsubu_even.shape[:2]:
        raise ValueError("bsubs and parity bsub shapes mismatch")

    ns, ntheta, nzeta = bsubu_even.shape
    nt2 = int(trig.ntheta2)
    if ntheta < nt2:
        raise ValueError("bsubu grid smaller than ntheta2")

    mmax = int(mmax_force)
    nmax = int(nmax_force)
    if mmax < 0 or nmax < 0:
        z = np.zeros((ns, nt2, nzeta), dtype=float)
        return z.copy(), z.copy(), z.copy(), z.copy()

    cosmui = np.asarray(trig.cosmui, dtype=acc_dtype)[:nt2, : mmax + 1]
    sinmui = np.asarray(trig.sinmui, dtype=acc_dtype)[:nt2, : mmax + 1]
    cosmu = np.asarray(trig.cosmu, dtype=acc_dtype)[:nt2, : mmax + 1]
    sinmu = np.asarray(trig.sinmu, dtype=acc_dtype)[:nt2, : mmax + 1]
    cosmum = np.asarray(trig.cosmum, dtype=acc_dtype)[:nt2, : mmax + 1]
    sinmum = np.asarray(trig.sinmum, dtype=acc_dtype)[:nt2, : mmax + 1]
    cosnv = np.asarray(trig.cosnv, dtype=acc_dtype)[:, : nmax + 1]
    sinnv = np.asarray(trig.sinnv, dtype=acc_dtype)[:, : nmax + 1]
    cosnvn = np.asarray(trig.cosnvn, dtype=acc_dtype)[:, : nmax + 1]
    sinnvn = np.asarray(trig.sinnvn, dtype=acc_dtype)[:, : nmax + 1]

    r0scale = float(getattr(trig, "r0scale", 1.0))
    base_dnorm = acc_dtype(1.0 / (r0scale**2))
    mnyq, nnyq = _jxbforce_nyquist_limits(trig)

    s_full = np.asarray(s, dtype=acc_dtype)
    if s_full.shape[0] < 2:
        pshalf = np.sqrt(np.maximum(s_full, 0.0))
    else:
        sh = 0.5 * (s_full[1:] + s_full[:-1])
        pshalf = np.concatenate([sh[:1], sh], axis=0)
        pshalf = np.sqrt(np.maximum(pshalf, 0.0))
    if pshalf.shape[0] > 1:
        pshalf[0] = pshalf[1]

    bsubu_out = np.zeros((ns, nt2, nzeta), dtype=acc_dtype)
    bsubv_out = np.zeros((ns, nt2, nzeta), dtype=acc_dtype)
    bsubsu = np.zeros((ns, nt2, nzeta), dtype=acc_dtype)
    bsubsv = np.zeros((ns, nt2, nzeta), dtype=acc_dtype)

    for js in range(ns):
        bsubs_s = bsubs[js, :nt2, :]
        bu_even = bsubu_even[js, :nt2, :]
        bu_odd = bsubu_odd[js, :nt2, :]
        bv_even = bsubv_even[js, :nt2, :]
        bv_odd = bsubv_odd[js, :nt2, :]
        for m in range(mmax + 1):
            use_odd = (m % 2) == 1
            bu_in = bu_odd if use_odd else bu_even
            bv_in = bv_odd if use_odd else bv_even
            for n in range(nmax + 1):
                dnorm1 = base_dnorm
                if mnyq > 0 and m == mnyq:
                    dnorm1 *= 0.5
                if nnyq > 0 and n == nnyq and n != 0:
                    dnorm1 *= 0.5
                if use_odd:
                    dnorm1 = dnorm1 / pshalf[js]

                bsubsmn1 = acc_dtype(0.0)
                bsubsmn2 = acc_dtype(0.0)
                bsubumn1 = acc_dtype(0.0)
                bsubumn2 = acc_dtype(0.0)
                bsubvmn1 = acc_dtype(0.0)
                bsubvmn2 = acc_dtype(0.0)

                for k in range(nzeta):
                    for j in range(nt2):
                        tsini1 = sinmui[j, m] * cosnv[k, n] * dnorm1
                        tsini2 = cosmui[j, m] * sinnv[k, n] * dnorm1
                        tcosi1 = cosmui[j, m] * cosnv[k, n] * dnorm1
                        tcosi2 = sinmui[j, m] * sinnv[k, n] * dnorm1
                        vbs = bsubs_s[j, k]
                        vu = bu_in[j, k]
                        vv = bv_in[j, k]
                        bsubsmn1 += tsini1 * vbs
                        bsubsmn2 += tsini2 * vbs
                        bsubumn1 += tcosi1 * vu
                        bsubumn2 += tcosi2 * vu
                        bsubvmn1 += tcosi1 * vv
                        bsubvmn2 += tcosi2 * vv

                for k in range(nzeta):
                    for j in range(nt2):
                        tcos1 = cosmu[j, m] * cosnv[k, n]
                        tcos2 = sinmu[j, m] * sinnv[k, n]
                        bsubu_out[js, j, k] += tcos1 * bsubumn1 + tcos2 * bsubumn2
                        bsubv_out[js, j, k] += tcos1 * bsubvmn1 + tcos2 * bsubvmn2

                        tcosm1 = cosmum[j, m] * cosnv[k, n]
                        tcosm2 = sinmum[j, m] * sinnv[k, n]
                        bsubsu[js, j, k] += tcosm1 * bsubsmn1 + tcosm2 * bsubsmn2
                        tcosn1 = sinmu[j, m] * sinnvn[k, n]
                        tcosn2 = cosmu[j, m] * cosnvn[k, n]
                        bsubsv[js, j, k] += tcosn1 * bsubsmn1 + tcosn2 * bsubsmn2

    return (
        np.asarray(bsubu_out, dtype=float),
        np.asarray(bsubv_out, dtype=float),
        np.asarray(bsubsu, dtype=float),
        np.asarray(bsubsv, dtype=float),
    )


def _filter_bsubuv_jxbforce_loop(
    *,
    bsubu: np.ndarray,
    bsubv: np.ndarray,
    trig,
    mmax_force: int,
    nmax_force: int,
    s: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Loop-based jxbforce low-pass filter (matches VMEC summation order)."""
    # Accumulate in long double for cancellation-sensitive mode sums.
    acc_dtype = np.longdouble
    bsubu = np.asarray(bsubu, dtype=acc_dtype)
    bsubv = np.asarray(bsubv, dtype=acc_dtype)
    ns, ntheta, nzeta = bsubu.shape

    nt2 = int(trig.ntheta2)
    if ntheta < nt2:
        raise ValueError("bsubu grid smaller than ntheta2")

    mmax = int(mmax_force)
    nmax = int(nmax_force)
    if mmax < 0 or nmax < 0:
        return bsubu[:, :nt2, :].copy(), bsubv[:, :nt2, :].copy()

    cosmui = np.asarray(trig.cosmui, dtype=acc_dtype)[:nt2, : mmax + 1]
    sinmui = np.asarray(trig.sinmui, dtype=acc_dtype)[:nt2, : mmax + 1]
    cosmu = np.asarray(trig.cosmu, dtype=acc_dtype)[:nt2, : mmax + 1]
    sinmu = np.asarray(trig.sinmu, dtype=acc_dtype)[:nt2, : mmax + 1]
    cosnv = np.asarray(trig.cosnv, dtype=acc_dtype)[:, : nmax + 1]
    sinnv = np.asarray(trig.sinnv, dtype=acc_dtype)[:, : nmax + 1]

    r0scale = float(getattr(trig, "r0scale", 1.0))
    base_dnorm = acc_dtype(1.0) / acc_dtype(r0scale**2)
    mnyq, nnyq = _jxbforce_nyquist_limits(trig)

    def _pshalf_from_s(s_full: np.ndarray) -> np.ndarray:
        if s_full.shape[0] < 2:
            return np.sqrt(np.maximum(s_full, 0.0))
        sh = 0.5 * (s_full[1:] + s_full[:-1])
        p = np.concatenate([sh[:1], sh], axis=0)
        return np.sqrt(np.maximum(p, 0.0))

    pshalf = None
    if s is not None:
        s_full = np.asarray(s, dtype=acc_dtype)
        pshalf = _pshalf_from_s(s_full)
        if pshalf.shape[0] > 1:
            pshalf[0] = pshalf[1]

    # If the requested filter spans the full available basis, return the
    # original fields to avoid introducing numerical drift.
    full_mmax, full_nmax = _jxbforce_nyquist_limits(trig)
    if mmax >= full_mmax and nmax >= full_nmax:
        return bsubu[:, :nt2, :].copy(), bsubv[:, :nt2, :].copy()

    bsubu_out = np.zeros((ns, nt2, nzeta), dtype=acc_dtype)
    bsubv_out = np.zeros((ns, nt2, nzeta), dtype=acc_dtype)

    for js in range(ns):
        bsubu_in = bsubu[js, :nt2, :]
        bsubv_in = bsubv[js, :nt2, :]
        for m in range(mmax + 1):
            for n in range(nmax + 1):
                dnorm1 = base_dnorm
                if mnyq > 0 and m == mnyq:
                    dnorm1 *= 0.5
                if nnyq > 0 and n == nnyq and n != 0:
                    dnorm1 *= 0.5
                # Undo odd-m sqrt(s) scaling (VMEC shalf factor).
                if (m % 2 == 1) and (pshalf is not None):
                    dnorm1 = dnorm1 / float(pshalf[js])

                bsubumn1 = acc_dtype(0.0)
                bsubumn2 = acc_dtype(0.0)
                bsubvmn1 = acc_dtype(0.0)
                bsubvmn2 = acc_dtype(0.0)

                for k in range(nzeta):
                    for j in range(nt2):
                        tsini1 = sinmui[j, m] * cosnv[k, n] * dnorm1
                        tsini2 = cosmui[j, m] * sinnv[k, n] * dnorm1
                        tcosi1 = cosmui[j, m] * cosnv[k, n] * dnorm1
                        tcosi2 = sinmui[j, m] * sinnv[k, n] * dnorm1
                        val_u = bsubu_in[j, k]
                        val_v = bsubv_in[j, k]
                        bsubumn1 += tcosi1 * val_u
                        bsubumn2 += tcosi2 * val_u
                        bsubvmn1 += tcosi1 * val_v
                        bsubvmn2 += tcosi2 * val_v

                for k in range(nzeta):
                    for j in range(nt2):
                        tcos1 = cosmu[j, m] * cosnv[k, n]
                        tcos2 = sinmu[j, m] * sinnv[k, n]
                        bsubu_out[js, j, k] += tcos1 * bsubumn1 + tcos2 * bsubumn2
                        bsubv_out[js, j, k] += tcos1 * bsubvmn1 + tcos2 * bsubvmn2

    return np.asarray(bsubu_out, dtype=float), np.asarray(bsubv_out, dtype=float)


def _filter_bsubuv_jxbforce_lasym_loop(
    *,
    bsubu: np.ndarray,
    bsubv: np.ndarray,
    trig,
    mmax_force: int,
    nmax_force: int,
    s: np.ndarray | None = None,
    bsubu_even: np.ndarray | None = None,
    bsubu_odd: np.ndarray | None = None,
    bsubv_even: np.ndarray | None = None,
    bsubv_odd: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Loop-accurate LASYM low-pass filter for bsubu/bsubv (jxbforce + fext_fft).

    Mirrors jxbforce.f for LASYM runs:
    1) contract full-grid fields to reduced-grid symmetric/antisymmetric channels
       (fsym_fft parity split),
    2) low-pass Fourier transform/inverse on the reduced grid,
    3) extend filtered channels back to full theta grid (fext_fft).
    """
    acc_dtype = np.longdouble
    bsubu = np.asarray(bsubu, dtype=acc_dtype)
    bsubv = np.asarray(bsubv, dtype=acc_dtype)
    if bsubu.shape != bsubv.shape:
        raise ValueError("bsubu/bsubv shape mismatch")
    have_parity_channels = (
        (bsubu_even is not None) and (bsubu_odd is not None) and (bsubv_even is not None) and (bsubv_odd is not None)
    )
    if have_parity_channels:
        bsubu_even = np.asarray(bsubu_even, dtype=acc_dtype)
        bsubu_odd = np.asarray(bsubu_odd, dtype=acc_dtype)
        bsubv_even = np.asarray(bsubv_even, dtype=acc_dtype)
        bsubv_odd = np.asarray(bsubv_odd, dtype=acc_dtype)
        if (
            bsubu_even.shape != bsubu.shape
            or bsubu_odd.shape != bsubu.shape
            or bsubv_even.shape != bsubu.shape
            or bsubv_odd.shape != bsubu.shape
        ):
            raise ValueError("LASYM bsub parity channel shape mismatch")

    ns, ntheta, nzeta = bsubu.shape
    nt2 = int(trig.ntheta2)
    nt1 = int(trig.ntheta1)
    nt3 = int(getattr(trig, "ntheta3", nt2))
    if ntheta < nt3:
        raise ValueError("LASYM bsubu grid smaller than ntheta3")

    mmax = int(mmax_force)
    nmax = int(nmax_force)
    if mmax < 0 or nmax < 0:
        return bsubu[:, :nt3, :].astype(float), bsubv[:, :nt3, :].astype(float)

    cosmui = np.asarray(trig.cosmui, dtype=acc_dtype)[:nt2, : mmax + 1]
    sinmui = np.asarray(trig.sinmui, dtype=acc_dtype)[:nt2, : mmax + 1]
    cosmu = np.asarray(trig.cosmu, dtype=acc_dtype)[:nt2, : mmax + 1]
    sinmu = np.asarray(trig.sinmu, dtype=acc_dtype)[:nt2, : mmax + 1]
    cosnv = np.asarray(trig.cosnv, dtype=acc_dtype)[:, : nmax + 1]
    sinnv = np.asarray(trig.sinnv, dtype=acc_dtype)[:, : nmax + 1]

    r0scale = float(getattr(trig, "r0scale", 1.0))
    base_dnorm = acc_dtype(1.0) / acc_dtype(r0scale**2)
    mnyq, nnyq = _jxbforce_nyquist_limits(trig)

    def _pshalf_from_s(s_full: np.ndarray) -> np.ndarray:
        if s_full.shape[0] < 2:
            return np.sqrt(np.maximum(s_full, 0.0))
        sh = 0.5 * (s_full[1:] + s_full[:-1])
        p = np.concatenate([sh[:1], sh], axis=0)
        return np.sqrt(np.maximum(p, 0.0))

    pshalf = None
    if s is not None:
        s_full = np.asarray(s, dtype=acc_dtype)
        pshalf = _pshalf_from_s(s_full)
        if pshalf.shape[0] > 1:
            pshalf[0] = pshalf[1]

    bsubu_out = np.zeros((ns, nt3, nzeta), dtype=acc_dtype)
    bsubv_out = np.zeros((ns, nt3, nzeta), dtype=acc_dtype)

    for js in range(ns):
        # Fortran fext/fsym paths use (zeta, theta) ordering.
        bu = np.asarray(bsubu[js, :nt3, :], dtype=acc_dtype).T  # (nzeta, ntheta3)
        bv = np.asarray(bsubv[js, :nt3, :], dtype=acc_dtype).T

        if have_parity_channels:
            bu0 = np.asarray(bsubu_even[js, :nt3, :], dtype=acc_dtype).T
            bv0 = np.asarray(bsubv_even[js, :nt3, :], dtype=acc_dtype).T
            if pshalf is not None:
                sh = acc_dtype(pshalf[js]) if pshalf[js] != 0.0 else acc_dtype(1.0)
            else:
                sh = acc_dtype(1.0)
            # VMEC stores odd channel as shalf*bsub*_odd before the immediate
            # in-loop divide by shalf in jxbforce.
            bu1 = np.asarray(bsubu_odd[js, :nt3, :], dtype=acc_dtype).T * sh
            bv1 = np.asarray(bsubv_odd[js, :nt3, :], dtype=acc_dtype).T * sh
            bu_ch = np.stack([bu0, bu1], axis=-1)
            bv_ch = np.stack([bv0, bv1], axis=-1)
        else:
            # Fallback path when only full bsub fields are available.
            bu_ch = np.stack([bu, bu], axis=-1)  # (nzeta, ntheta3, 2)
            bv_ch = np.stack([bv, bv], axis=-1)

        bu_s = np.zeros((nzeta, nt2, 2), dtype=acc_dtype)
        bu_a = np.zeros((nzeta, nt2, 2), dtype=acc_dtype)
        bv_s = np.zeros((nzeta, nt2, 2), dtype=acc_dtype)
        bv_a = np.zeros((nzeta, nt2, 2), dtype=acc_dtype)

        # fsym_fft contraction.
        for i in range(nt2):
            ir = 0 if i == 0 else (nt1 - i)
            for kz in range(nzeta):
                kzr = 0 if kz == 0 else (nzeta - kz)
                bu_a[kz, i, :] = 0.5 * (bu_ch[kz, i, :] - bu_ch[kzr, ir, :])
                bu_s[kz, i, :] = 0.5 * (bu_ch[kz, i, :] + bu_ch[kzr, ir, :])
                bv_a[kz, i, :] = 0.5 * (bv_ch[kz, i, :] - bv_ch[kzr, ir, :])
                bv_s[kz, i, :] = 0.5 * (bv_ch[kz, i, :] + bv_ch[kzr, ir, :])

        bsubua = np.zeros((nzeta, nt2, 2), dtype=acc_dtype)
        bsubva = np.zeros((nzeta, nt2, 2), dtype=acc_dtype)

        for m in range(mmax + 1):
            mparity = m & 1
            for n in range(nmax + 1):
                dnorm1 = base_dnorm
                if mnyq > 0 and m == mnyq:
                    dnorm1 *= 0.5
                if nnyq > 0 and n == nnyq and n != 0:
                    dnorm1 *= 0.5

                bsubumn1 = acc_dtype(0.0)
                bsubumn2 = acc_dtype(0.0)
                bsubvmn1 = acc_dtype(0.0)
                bsubvmn2 = acc_dtype(0.0)
                bsubumn3 = acc_dtype(0.0)
                bsubumn4 = acc_dtype(0.0)
                bsubvmn3 = acc_dtype(0.0)
                bsubvmn4 = acc_dtype(0.0)

                for k in range(nzeta):
                    for j in range(nt2):
                        tsini1 = sinmui[j, m] * cosnv[k, n] * dnorm1
                        tsini2 = cosmui[j, m] * sinnv[k, n] * dnorm1
                        tcosi1 = cosmui[j, m] * cosnv[k, n] * dnorm1
                        tcosi2 = sinmui[j, m] * sinnv[k, n] * dnorm1

                        bsubumn1 += tcosi1 * bu_s[k, j, mparity]
                        bsubumn2 += tcosi2 * bu_s[k, j, mparity]
                        bsubvmn1 += tcosi1 * bv_s[k, j, mparity]
                        bsubvmn2 += tcosi2 * bv_s[k, j, mparity]

                        bsubumn3 += tsini1 * bu_a[k, j, mparity]
                        bsubumn4 += tsini2 * bu_a[k, j, mparity]
                        bsubvmn3 += tsini1 * bv_a[k, j, mparity]
                        bsubvmn4 += tsini2 * bv_a[k, j, mparity]

                for k in range(nzeta):
                    for j in range(nt2):
                        tcos1 = cosmu[j, m] * cosnv[k, n]
                        tcos2 = sinmu[j, m] * sinnv[k, n]
                        bsubua[k, j, 0] += tcos1 * bsubumn1 + tcos2 * bsubumn2
                        bsubva[k, j, 0] += tcos1 * bsubvmn1 + tcos2 * bsubvmn2

                        tsin1 = sinmu[j, m] * cosnv[k, n]
                        tsin2 = cosmu[j, m] * sinnv[k, n]
                        bsubua[k, j, 1] += tsin1 * bsubumn3 + tsin2 * bsubumn4
                        bsubva[k, j, 1] += tsin1 * bsubvmn3 + tsin2 * bsubvmn4

        # fext_fft extension to full theta grid.
        bu_full = np.zeros((nzeta, nt3), dtype=acc_dtype)
        bv_full = np.zeros((nzeta, nt3), dtype=acc_dtype)
        bu_full[:, :nt2] = bsubua[:, :, 0] + bsubua[:, :, 1]
        bv_full[:, :nt2] = bsubva[:, :, 0] + bsubva[:, :, 1]
        for i in range(nt2, nt3):
            ir = nt1 - i
            for kz in range(nzeta):
                kzr = 0 if kz == 0 else (nzeta - kz)
                bu_full[kz, i] = bsubua[kzr, ir, 0] - bsubua[kzr, ir, 1]
                bv_full[kz, i] = bsubva[kzr, ir, 0] - bsubva[kzr, ir, 1]

        bsubu_out[js, :, :] = bu_full.T
        bsubv_out[js, :, :] = bv_full.T

    return np.asarray(bsubu_out, dtype=float), np.asarray(bsubv_out, dtype=float)


def _jxbforce_bsubsu_bsubsv_loop(
    *,
    bsubs: np.ndarray,
    trig,
    mmax_force: int,
    nmax_force: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Loop-based bsubsu/bsubsv reconstruction (jxbforce)."""
    # Cancellation in jxbforce transforms is severe for some equilibria
    # (e.g. QI_nfp2 near edge). Accumulate in long double, then cast back.
    acc_dtype = np.longdouble
    bsubs = np.asarray(bsubs, dtype=acc_dtype)
    ns, ntheta, nzeta = bsubs.shape
    nt2 = int(trig.ntheta2)
    if ntheta < nt2:
        raise ValueError("bsubs grid smaller than ntheta2")

    mmax = int(mmax_force)
    nmax = int(nmax_force)
    if mmax < 0 or nmax < 0:
        return np.zeros((ns, nt2, nzeta), dtype=float), np.zeros((ns, nt2, nzeta), dtype=float)

    cosmui = np.asarray(trig.cosmui, dtype=acc_dtype)[:nt2, : mmax + 1]
    sinmui = np.asarray(trig.sinmui, dtype=acc_dtype)[:nt2, : mmax + 1]
    cosmu = np.asarray(trig.cosmu, dtype=acc_dtype)[:nt2, : mmax + 1]
    sinmu = np.asarray(trig.sinmu, dtype=acc_dtype)[:nt2, : mmax + 1]
    cosmum = np.asarray(trig.cosmum, dtype=acc_dtype)[:nt2, : mmax + 1]
    sinmum = np.asarray(trig.sinmum, dtype=acc_dtype)[:nt2, : mmax + 1]
    cosnv = np.asarray(trig.cosnv, dtype=acc_dtype)[:, : nmax + 1]
    sinnv = np.asarray(trig.sinnv, dtype=acc_dtype)[:, : nmax + 1]
    cosnvn = np.asarray(trig.cosnvn, dtype=acc_dtype)[:, : nmax + 1]
    sinnvn = np.asarray(trig.sinnvn, dtype=acc_dtype)[:, : nmax + 1]

    r0scale = float(getattr(trig, "r0scale", 1.0))
    base_dnorm = acc_dtype(1.0 / (r0scale**2))
    mnyq, nnyq = _jxbforce_nyquist_limits(trig)

    bsubsu = np.zeros((ns, nt2, nzeta), dtype=acc_dtype)
    bsubsv = np.zeros((ns, nt2, nzeta), dtype=acc_dtype)

    for js in range(ns):
        bsubs_s = bsubs[js, :nt2, :]
        for m in range(mmax + 1):
            for n in range(nmax + 1):
                dnorm1 = base_dnorm
                if mnyq > 0 and m == mnyq:
                    dnorm1 *= 0.5
                if nnyq > 0 and n == nnyq and n != 0:
                    dnorm1 *= 0.5

                bsubsmn1 = acc_dtype(0.0)
                bsubsmn2 = acc_dtype(0.0)
                for k in range(nzeta):
                    for j in range(nt2):
                        tsini1 = sinmui[j, m] * cosnv[k, n] * dnorm1
                        tsini2 = cosmui[j, m] * sinnv[k, n] * dnorm1
                        val = bsubs_s[j, k]
                        bsubsmn1 += tsini1 * val
                        bsubsmn2 += tsini2 * val

                for k in range(nzeta):
                    for j in range(nt2):
                        tcosm1 = cosmum[j, m] * cosnv[k, n]
                        tcosm2 = sinmum[j, m] * sinnv[k, n]
                        bsubsu[js, j, k] += tcosm1 * bsubsmn1 + tcosm2 * bsubsmn2
                        tcosn1 = sinmu[j, m] * sinnvn[k, n]
                        tcosn2 = cosmu[j, m] * cosnvn[k, n]
                        bsubsv[js, j, k] += tcosn1 * bsubsmn1 + tcosn2 * bsubsmn2

    return np.asarray(bsubsu, dtype=float), np.asarray(bsubsv, dtype=float)


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


def _compute_mercier(
    *,
    state: VMECState,
    geom_modes,
    s: np.ndarray,
    lconm1: bool,
    lthreed: bool,
    lasym: bool,
    nfp: int,
    lbsubs: bool,
    mmax_force: int,
    nmax_force: int,
    pres: np.ndarray,
    vp: np.ndarray,
    phips: np.ndarray,
    iotas: np.ndarray,
    bsq: np.ndarray,
    sqrtg: np.ndarray,
    bsubu: np.ndarray,
    bsubv: np.ndarray,
    bsubu_parity_even: np.ndarray | None = None,
    bsubu_parity_odd: np.ndarray | None = None,
    bsubv_parity_even: np.ndarray | None = None,
    bsubv_parity_odd: np.ndarray | None = None,
    bsupu: np.ndarray,
    bsupv: np.ndarray,
    trig,
    geom: dict[str, Any],
    jac_half: Any | None = None,
    force_rs: np.ndarray | None = None,
    force_zs: np.ndarray | None = None,
    force_ru12: np.ndarray | None = None,
    force_zu12: np.ndarray | None = None,
    sigma_an: np.ndarray | None = None,
    bsubu_raw: np.ndarray | None = None,
    bsubv_raw: np.ndarray | None = None,
    signgs: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Port of VMEC mercier.f to compute Mercier and jxbforce-style scalars."""
    ns = int(pres.shape[0])
    if ns < 3:
        zero = np.zeros((ns,), dtype=float)
        return tuple(zero.copy() for _ in range(8))
    hs = 1.0 / float(ns - 1)

    sqrtg = np.asarray(sqrtg, dtype=float)
    if sigma_an is None:
        sigma_an = np.ones_like(sqrtg)
    else:
        sigma_an = np.asarray(sigma_an, dtype=float)
    if signgs is not None and int(signgs) != 0:
        sign_jac = float(np.sign(float(signgs)))
    else:
        sign_jac = float(np.sign(sqrtg[-1, 0, 0])) if float(sqrtg[-1, 0, 0]) != 0.0 else 1.0
    phip_real = (2.0 * np.pi) * np.asarray(phips, dtype=float) * sign_jac
    vp_real = np.zeros_like(phip_real)
    vp_real[1:] = sign_jac * (2.0 * np.pi) ** 2 * np.asarray(vp[1:], dtype=float) / phip_real[1:]

    wint = _vmec_wint_from_trig(trig)
    nzeta = wint.shape[1]
    ntheta = wint.shape[0]
    exact_sum = os.getenv("VMEC_JAX_MERCIER_EXACT_SUM", "").strip().lower() not in ("", "0", "false", "no")
    if exact_sum:
        # Match VMEC2000 Fortran summation order on cancellation-sensitive
        # surface averages (slow; intended for parity debugging).
        def _sum_w(arr: np.ndarray) -> float:
            acc = 0.0
            for j in range(ntheta):
                wrow = wint[j]
                arow = arr[j]
                for k in range(nzeta):
                    acc += float(arow[k]) * float(wrow[k])
            return acc
    else:
        wint_f = np.asarray(wint, dtype=float)

        def _sum_w(arr: np.ndarray) -> float:
            # Fast path: vectorized weighted sum on the reduced (theta,zeta) grid.
            return float(np.einsum("ij,ij->", np.asarray(arr, dtype=float), wint_f, optimize=True))

    # Geometry fields on the full mesh (physical values from internal VMEC coefficients).
    R = np.asarray(geom["R"], dtype=float)
    Z = np.asarray(geom["Z"], dtype=float)
    Ru = np.asarray(geom["Ru"], dtype=float)
    Zu = np.asarray(geom["Zu"], dtype=float)
    Rv = np.asarray(geom["Rv"], dtype=float)
    Zv = np.asarray(geom["Zv"], dtype=float)

    bsubs = _compute_bsubs_half_mesh(
        state=state,
        geom_modes=geom_modes,
        s=s,
        lconm1=bool(lconm1),
        lthreed=bool(lthreed),
        lasym=bool(lasym),
        bsupu=bsupu,
        bsupv=bsupv,
        trig=trig,
        geom=geom,
        jac_half=jac_half,
        force_rs=force_rs,
        force_zs=force_zs,
        force_ru12=force_ru12,
        force_zu12=force_zu12,
        apply_scalxc=os.getenv("VMEC_JAX_BSS_APPLY_SCALXC", "1") not in ("", "0"),
    )

    # Preserve the full bsubu/bsubv inputs for optional jxbout parity dumps.
    bsubu_full = np.asarray(bsubu, dtype=float)
    bsubv_full = np.asarray(bsubv, dtype=float)

    # LASYM: keep the full-grid bsubu/bsubv supplied by bcovar. VMEC's
    # jxbforce/mercier operate on the full (ntheta3) mesh after parity
    # extension, so avoid re-splitting unless we are explicitly given a
    # reduced-grid field.
    if bool(lasym):
        bsubu = np.asarray(bsubu, dtype=float)
        bsubv = np.asarray(bsubv, dtype=float)
        nt2 = int(trig.ntheta2)
        nt3 = int(getattr(trig, "ntheta3", nt2))
        if bsubu.shape[1] == nt2 and nt3 > nt2:
            bsubu = _vmec_symoutput_expand(sym=bsubu, asym=None, trig=trig)
        if bsubv.shape[1] == nt2 and nt3 > nt2:
            bsubv = _vmec_symoutput_expand(sym=bsubv, asym=None, trig=trig)

    # For lasym=False, VMEC stores fields on the reduced theta grid already.
    # Only apply symmetrization when a full theta grid is supplied.
    if not bool(lasym):
        nt2 = int(trig.ntheta2)
        nt1 = int(trig.ntheta1)
        if nt2 > 0:
            bsubu = np.asarray(bsubu, dtype=float)
            bsubv = np.asarray(bsubv, dtype=float)
            if bsubu.shape[1] > nt2:
                i0 = np.arange(nt2, dtype=int)
                ir0 = np.where(i0 == 0, 0, nt1 - i0)
                kk = (nzeta - np.arange(nzeta, dtype=int)) % nzeta

                def _sym_half(arr: np.ndarray) -> np.ndarray:
                    a_half = arr[:, :nt2, :]
                    a_ref = arr[:, ir0, :][:, :, kk]
                    return 0.5 * (a_half + a_ref)

                bsubu = _sym_half(bsubu)
                bsubv = _sym_half(bsubv)
            else:
                bsubu = bsubu[:, :nt2, :]
                bsubv = bsubv[:, :nt2, :]
    else:
        # Match jxbforce LASYM preprocessing: low-pass filter bsubu/bsubv on the
        # reduced grid then extend back to full theta before Mercier assembly.
        if os.getenv("VMEC_JAX_MERCIER_LASYM_FILTER", "1") not in ("", "0"):
            use_parity_channels = os.getenv("VMEC_JAX_LASYM_FILTER_USE_PARITY_CHANNELS", "0") not in (
                "",
                "0",
                "false",
                "no",
            )
            bsubu, bsubv = _filter_bsubuv_jxbforce_lasym_loop(
                bsubu=np.asarray(bsubu, dtype=float),
                bsubv=np.asarray(bsubv, dtype=float),
                trig=trig,
                mmax_force=max(int(mmax_force), 0),
                nmax_force=max(int(nmax_force), 0),
                s=np.asarray(s, dtype=float),
                bsubu_even=None
                if (not use_parity_channels) or (bsubu_parity_even is None)
                else np.asarray(bsubu_parity_even, dtype=float),
                bsubu_odd=None
                if (not use_parity_channels) or (bsubu_parity_odd is None)
                else np.asarray(bsubu_parity_odd, dtype=float),
                bsubv_even=None
                if (not use_parity_channels) or (bsubv_parity_even is None)
                else np.asarray(bsubv_parity_even, dtype=float),
                bsubv_odd=None
                if (not use_parity_channels) or (bsubv_parity_odd is None)
                else np.asarray(bsubv_parity_odd, dtype=float),
            )

    # Parity-decomposed geometry (totzsps convention): X = X_even + sqrt(s)*X_odd.
    from .vmec_jacobian import _apply_vmec_axis_rules
    from .vmec_realspace import (
        vmec_realspace_synthesis,
        vmec_realspace_synthesis_dtheta,
    )

    m = np.asarray(geom_modes.m)
    mask_even = (m % 2) == 0
    mask_odd = np.logical_not(mask_even)
    Rcos = np.asarray(state.Rcos)
    Rsin = np.asarray(state.Rsin)
    Zcos = np.asarray(state.Zcos)
    Zsin = np.asarray(state.Zsin)
    if bool(lconm1):
        Rcos, Zsin, Rsin, Zcos = vmec_m1_internal_to_physical_signed(
            Rcos=Rcos,
            Zsin=Zsin,
            Rsin=Rsin,
            Zcos=Zcos,
            modes=geom_modes,
            lthreed=bool(lthreed),
            lasym=bool(lasym),
            lconm1=bool(lconm1),
        )
    Rcos = _apply_vmec_axis_rules(Rcos, m)
    Rsin = _apply_vmec_axis_rules(Rsin, m)
    Zcos = _apply_vmec_axis_rules(Zcos, m)
    Zsin = _apply_vmec_axis_rules(Zsin, m)

    coeff_cos_stack = np.stack([Rcos, Zcos], axis=0)
    coeff_sin_stack = np.stack([Rsin, Zsin], axis=0)
    mask_stack = np.stack([mask_even.astype(float), mask_odd.astype(float)], axis=0)
    coeff_cos = coeff_cos_stack[None, ...] * mask_stack[:, None, None, :]
    coeff_sin = coeff_sin_stack[None, ...] * mask_stack[:, None, None, :]

    stack = vmec_realspace_synthesis(
        coeff_cos=coeff_cos,
        coeff_sin=coeff_sin,
        modes=geom_modes,
        trig=trig,
        coeffs_internal=True,
        apply_scalxc=True,
        s=s,
    )
    stack_t = vmec_realspace_synthesis_dtheta(
        coeff_cos=coeff_cos,
        coeff_sin=coeff_sin,
        modes=geom_modes,
        trig=trig,
        coeffs_internal=True,
        apply_scalxc=True,
        s=s,
    )
    stack_p = vmec_realspace_synthesis_dzeta_phys(
        coeff_cos=coeff_cos,
        coeff_sin=coeff_sin,
        modes=geom_modes,
        trig=trig,
        coeffs_internal=True,
        apply_scalxc=True,
        s=s,
    )
    even = np.asarray(stack[0])
    odd = np.asarray(stack[1])
    even_t = np.asarray(stack_t[0])
    odd_t = np.asarray(stack_t[1])
    even_p = np.asarray(stack_p[0])
    odd_p = np.asarray(stack_p[1])

    R_even = even[0]
    R_odd = odd[0]
    Z_even = even[1]
    Z_odd = odd[1]
    Ru_even = even_t[0]
    Ru_odd = odd_t[0]
    Zu_even = even_t[1]
    Zu_odd = odd_t[1]
    Rv_even = even_p[0]
    Rv_odd = odd_p[0]
    Zv_even = even_p[1]
    Zv_odd = odd_p[1]

    # VMEC jxbforce-style derivatives of bsubs on the reduced grid.
    mmax = int(mmax_force)
    nmax = int(nmax_force)
    nt2 = int(trig.ntheta2)
    nzeta = int(trig.cosnv.shape[0])

    bsubs_use = bsubs.copy()
    if ns > 2:
        # Match VMEC's jxbforce/wrout convention: average to the full mesh.
        bsubs_use[1:-1] = 0.5 * (bsubs_use[1:-1] + bsubs_use[2:])
    bsubs_use[0] = 0.0

    # Reconstruct bsubsu/bsubsv via Fourier coefficients to match jxbforce.
    # Optional strict path: run the low-pass bsubu/bsubv filter and bsubs
    # derivative reconstruction in a single coupled loop (Fortran-like order).
    use_coupled = (not bool(lasym)) and os.getenv("VMEC_JAX_JXBFORCE_COUPLED_BSUB", "0") not in (
        "",
        "0",
        "false",
        "no",
    )
    if use_coupled:
        psh = _pshalf_from_s(np.asarray(s, dtype=float))[:, None, None]
        if psh.shape[0] > 1:
            psh[0] = psh[1]
        use_bc_parity = os.getenv("VMEC_JAX_JXBFORCE_COUPLED_USE_BC_PARITY", "0") not in ("", "0", "false", "no")
        if use_bc_parity:
            bu_even = (
                np.asarray(bsubu_parity_even, dtype=float)
                if (bsubu_parity_even is not None and np.asarray(bsubu_parity_even).shape == np.asarray(bsubu).shape)
                else np.asarray(bsubu, dtype=float)
            )
            bv_even = (
                np.asarray(bsubv_parity_even, dtype=float)
                if (bsubv_parity_even is not None and np.asarray(bsubv_parity_even).shape == np.asarray(bsubv).shape)
                else np.asarray(bsubv, dtype=float)
            )
            bu_odd = (
                np.asarray(bsubu_parity_odd, dtype=float)
                if (bsubu_parity_odd is not None and np.asarray(bsubu_parity_odd).shape == np.asarray(bsubu).shape)
                else psh * bu_even
            )
            bv_odd = (
                np.asarray(bsubv_parity_odd, dtype=float)
                if (bsubv_parity_odd is not None and np.asarray(bsubv_parity_odd).shape == np.asarray(bsubv).shape)
                else psh * bv_even
            )
        else:
            bu_even = np.asarray(bsubu, dtype=float)
            bv_even = np.asarray(bsubv, dtype=float)
            bu_odd = psh * bu_even
            bv_odd = psh * bv_even
        bsubu, bsubv, bsubsu, bsubsv = _jxbforce_filter_with_bsubs_derivs_loop(
            bsubs=bsubs_use,
            bsubu_even=bu_even,
            bsubu_odd=bu_odd,
            bsubv_even=bv_even,
            bsubv_odd=bv_odd,
            trig=trig,
            mmax_force=mmax,
            nmax_force=nmax,
            s=np.asarray(s, dtype=float),
        )
    # Default to the vectorized path for performance; the loop-based path is
    # retained for parity debugging.
    elif (not bool(lasym)) and os.getenv("VMEC_JAX_JXBFORCE_LOOP", "0") not in ("", "0"):
        bsubsu, bsubsv = _jxbforce_bsubsu_bsubsv_loop(
            bsubs=bsubs_use,
            trig=trig,
            mmax_force=mmax,
            nmax_force=nmax,
        )
    else:
        cosmui = np.asarray(trig.cosmui, dtype=float)[:nt2, : mmax + 1]
        sinmui = np.asarray(trig.sinmui, dtype=float)[:nt2, : mmax + 1]
        cosmu = np.asarray(trig.cosmu, dtype=float)[:nt2, : mmax + 1]
        sinmu = np.asarray(trig.sinmu, dtype=float)[:nt2, : mmax + 1]
        cosmum = np.asarray(trig.cosmum, dtype=float)[:nt2, : mmax + 1]
        sinmum = np.asarray(trig.sinmum, dtype=float)[:nt2, : mmax + 1]
        cosnv = np.asarray(trig.cosnv, dtype=float)[:, : nmax + 1]
        sinnv = np.asarray(trig.sinnv, dtype=float)[:, : nmax + 1]
        cosnvn = np.asarray(trig.cosnvn, dtype=float)[:, : nmax + 1]
        sinnvn = np.asarray(trig.sinnvn, dtype=float)[:, : nmax + 1]

        r0scale = float(getattr(trig, "r0scale", 1.0))
        dnorm1 = 1.0 / (r0scale**2)
        dmult = np.full((mmax + 1, nmax + 1), dnorm1, dtype=float)
        mnyq = np.asarray(trig.cosmui, dtype=float).shape[1] - 1
        nnyq = np.asarray(trig.cosnv, dtype=float).shape[1] - 1
        if mnyq > 0 and mnyq <= mmax:
            dmult[mnyq, :] *= 0.5
        if nnyq > 0 and nnyq <= nmax:
            dmult[:, nnyq] *= 0.5

        if bool(lasym):
            bsubs_sym, bsubs_asym = _vmec_symoutput_split(f=bsubs_use, trig=trig, reversed_sym=True)

            # Symmetric (sin) channel.
            f_theta_sin = np.einsum("sik,im->smk", bsubs_sym[:, :nt2, :], sinmui, optimize=True)
            f_theta_cos = np.einsum("sik,im->smk", bsubs_sym[:, :nt2, :], cosmui, optimize=True)
            bsubsmn1 = np.einsum("smk,kn->smn", f_theta_sin, cosnv, optimize=True) * dmult[None, :, :]
            bsubsmn2 = np.einsum("smk,kn->smn", f_theta_cos, sinnv, optimize=True) * dmult[None, :, :]

            tmp_su_1 = np.einsum("smn,im->sin", bsubsmn1, cosmum, optimize=True)
            tmp_su_2 = np.einsum("smn,im->sin", bsubsmn2, sinmum, optimize=True)
            bsubsu_s = np.einsum("sin,kn->sik", tmp_su_1, cosnv, optimize=True) + np.einsum(
                "sin,kn->sik", tmp_su_2, sinnv, optimize=True
            )

            tmp_sv_1 = np.einsum("smn,im->sin", bsubsmn1, sinmu, optimize=True)
            tmp_sv_2 = np.einsum("smn,im->sin", bsubsmn2, cosmu, optimize=True)
            bsubsv_s = np.einsum("sin,kn->sik", tmp_sv_1, sinnvn, optimize=True) + np.einsum(
                "sin,kn->sik", tmp_sv_2, cosnvn, optimize=True
            )

            # Asymmetric (cos) channel.
            f_theta_cos_a = np.einsum("sik,im->smk", bsubs_asym[:, :nt2, :], cosmui, optimize=True)
            f_theta_sin_a = np.einsum("sik,im->smk", bsubs_asym[:, :nt2, :], sinmui, optimize=True)
            bsubsmn3 = np.einsum("smk,kn->smn", f_theta_cos_a, cosnv, optimize=True) * dmult[None, :, :]
            bsubsmn4 = np.einsum("smk,kn->smn", f_theta_sin_a, sinnv, optimize=True) * dmult[None, :, :]

            tmp_su_3 = np.einsum("smn,im->sin", bsubsmn3, sinmum, optimize=True)
            tmp_su_4 = np.einsum("smn,im->sin", bsubsmn4, cosmum, optimize=True)
            bsubsu_a = np.einsum("sin,kn->sik", tmp_su_3, cosnv, optimize=True) + np.einsum(
                "sin,kn->sik", tmp_su_4, sinnv, optimize=True
            )

            tmp_sv_3 = np.einsum("smn,im->sin", bsubsmn3, cosmu, optimize=True)
            tmp_sv_4 = np.einsum("smn,im->sin", bsubsmn4, sinmu, optimize=True)
            bsubsv_a = np.einsum("sin,kn->sik", tmp_sv_3, sinnvn, optimize=True) + np.einsum(
                "sin,kn->sik", tmp_sv_4, cosnvn, optimize=True
            )

            # Extend parity-separated channels to the full theta grid.
            nt1 = int(trig.ntheta1)
            nt2 = int(trig.ntheta2)
            nt3 = int(getattr(trig, "ntheta3", nt2))
            nzeta = int(np.asarray(bsubsu_s).shape[2])

            def _extend_parity_to_full(par0: np.ndarray, par1: np.ndarray) -> np.ndarray:
                full = np.zeros((par0.shape[0], nt3, nzeta), dtype=float)
                full[:, :nt2, :] = par0 + par1
                if nt3 == nt2:
                    return full
                i0 = np.arange(nt2, dtype=int)
                ir0 = np.where(i0 == 0, 0, nt1 - i0)
                kk = (nzeta - np.arange(nzeta, dtype=int)) % nzeta
                mask = ir0 >= nt2
                if np.any(mask):
                    ir = ir0[mask]
                    ref0 = par0[:, mask, :][:, :, kk]
                    ref1 = par1[:, mask, :][:, :, kk]
                    full[:, ir, :] = ref0 - ref1
                return full

            bsubsu = _extend_parity_to_full(bsubsu_s, bsubsu_a)
            bsubsv = _extend_parity_to_full(bsubsv_s, bsubsv_a)
        else:
            # jxbforce-style Fourier coefficients for bsubs (sin basis).
            f_theta_sin = np.einsum("sik,im->smk", bsubs_use[:, :nt2, :], sinmui, optimize=True)
            f_theta_cos = np.einsum("sik,im->smk", bsubs_use[:, :nt2, :], cosmui, optimize=True)
            bsubsmn1 = np.einsum("smk,kn->smn", f_theta_sin, cosnv, optimize=True) * dmult[None, :, :]
            bsubsmn2 = np.einsum("smk,kn->smn", f_theta_cos, sinnv, optimize=True) * dmult[None, :, :]

            # Reconstruct bsubsu/bsubsv (jxbforce).
            tmp_su_1 = np.einsum("smn,im->sin", bsubsmn1, cosmum, optimize=True)
            tmp_su_2 = np.einsum("smn,im->sin", bsubsmn2, sinmum, optimize=True)
            bsubsu = np.einsum("sin,kn->sik", tmp_su_1, cosnv, optimize=True) + np.einsum(
                "sin,kn->sik", tmp_su_2, sinnv, optimize=True
            )

            tmp_sv_1 = np.einsum("smn,im->sin", bsubsmn1, sinmu, optimize=True)
            tmp_sv_2 = np.einsum("smn,im->sin", bsubsmn2, cosmu, optimize=True)
            bsubsv = np.einsum("sin,kn->sik", tmp_sv_1, sinnvn, optimize=True) + np.einsum(
                "sin,kn->sik", tmp_sv_2, cosnvn, optimize=True
            )

    bsubu = np.asarray(bsubu, dtype=float)
    bsubv = np.asarray(bsubv, dtype=float)
    if bool(lasym):
        if lbsubs:
            bsubs_use, bsubsu, bsubsv = _jxbforce_apply_bsubs_correction_lasym_true(
                bsubu=bsubu,
                bsubv=bsubv,
                bsubs=bsubs_use,
                bsubsu=bsubsu,
                bsubsv=bsubsv,
                bsupu=np.asarray(bsupu, dtype=float),
                bsupv=np.asarray(bsupv, dtype=float),
                sqrtg=np.asarray(sqrtg, dtype=float),
                pres=np.asarray(pres, dtype=float),
                vp=np.asarray(vp, dtype=float),
                hs=float(hs),
                signgs=float(sign_jac),
                trig=trig,
                nfp=int(nfp),
                sum_w=_sum_w,
            )
    elif lbsubs:
        bsubs_use, bsubsu, bsubsv = _jxbforce_apply_bsubs_correction_lasym_false(
            bsubu=bsubu,
            bsubv=bsubv,
            bsubs=bsubs_use,
            bsubsu=bsubsu,
            bsubsv=bsubsv,
            bsupu=np.asarray(bsupu, dtype=float),
            bsupv=np.asarray(bsupv, dtype=float),
            sqrtg=np.asarray(sqrtg, dtype=float),
            pres=np.asarray(pres, dtype=float),
            vp=np.asarray(vp, dtype=float),
            hs=float(hs),
            signgs=float(sign_jac),
            trig=trig,
            nfp=int(nfp),
            sum_w=_sum_w,
        )

    if os.getenv("VMEC_JAX_DUMP_JXB_CHANNELS", "") not in ("", "0"):
        from pathlib import Path

        tag = os.getenv("VMEC_JAX_DUMP_TAG", "").strip()
        dump_dir = Path(os.getenv("VMEC_JAX_DUMP_DIR", ".")).expanduser().resolve()
        dump_dir.mkdir(parents=True, exist_ok=True)
        dump_js = os.getenv("VMEC_JAX_DUMP_JS", "").strip()
        js_sel = int(dump_js) if dump_js not in ("",) else -1
        js_list = range(ns) if js_sel < 0 else [js_sel]
        out = {}
        for js in js_list:
            if js < 0 or js >= ns:
                continue
            key = f"js{js}"
            out[key] = dict(
                bsubs=np.asarray(bsubs_use[js], dtype=float),
                bsubsu0=np.asarray(bsubsu[js], dtype=float),
                bsubsv0=np.asarray(bsubsv[js], dtype=float),
            )
        np.savez(
            dump_dir / f"jxb_channels_wout{('_' + tag) if tag else ''}.npz",
            ntheta2=int(trig.ntheta2),
            ntheta3=int(getattr(trig, "ntheta3", trig.ntheta2)),
            nzeta=int(trig.cosnv.shape[0]),
            **out,
        )
    if os.getenv("VMEC_JAX_DUMP_JXBOUT", "") not in ("", "0"):
        from pathlib import Path

        tag = os.getenv("VMEC_JAX_DUMP_TAG", "").strip()
        dump_dir = Path(os.getenv("VMEC_JAX_DUMP_DIR", ".")).expanduser().resolve()
        dump_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            dump_dir / f"jxbout_jax{('_' + tag) if tag else ''}.npz",
            bsubs=np.asarray(bsubs_use, dtype=float),
            bsubu=np.asarray(bsubu, dtype=float),
            bsubv=np.asarray(bsubv, dtype=float),
            bsubu_full=np.asarray(bsubu_full, dtype=float),
            bsubv_full=np.asarray(bsubv_full, dtype=float),
            ntheta3=int(getattr(trig, "ntheta3", trig.ntheta2)),
            nzeta=int(trig.cosnv.shape[0]),
            ns=int(ns),
        )
    itheta = np.zeros_like(bsubs)
    izeta = np.zeros_like(bsubs)
    ohs = 1.0 / hs
    if ns > 2:
        itheta[1:-1] = bsubsv[1:-1] - ohs * (bsubv[2:] - bsubv[1:-1])
        izeta[1:-1] = -bsubsu[1:-1] + ohs * (bsubu[2:] - bsubu[1:-1])
        izeta[0] = 2.0 * izeta[1] - izeta[2]
        izeta[-1] = 2.0 * izeta[-2] - izeta[-3]
    # Match jxbforce: convert to MKS units (1/mu0 factor).
    mu0 = 4e-7 * np.pi
    itheta = itheta / mu0
    izeta = izeta / mu0
    bdotk = np.zeros_like(bsubs)
    if ns > 2:
        bsubu1 = 0.5 * (bsubu[2:] + bsubu[1:-1])
        bsubv1 = 0.5 * (bsubv[2:] + bsubv[1:-1])
        bdotk[1:-1] = itheta[1:-1] * bsubu1 + izeta[1:-1] * bsubv1

    # VMEC multiplies bdotk by mu0 before feeding Mercier (bdotj = sqrt(g)*J·B).
    bdotk_merc = MU0 * bdotk

    # Optional debug dump for parity work.
    if os.getenv("VMEC_JAX_DUMP_JDOTB", "") not in ("", "0"):
        from pathlib import Path

        tag = os.getenv("VMEC_JAX_DUMP_TAG", "").strip()
        name = "jdotb_debug" + (f"_{tag}" if tag else "") + ".npz"
        outdir = Path(os.getenv("VMEC_JAX_DUMP_DIR", ".")).expanduser().resolve()
        outdir.mkdir(parents=True, exist_ok=True)
        np.savez(
            outdir / name,
            bdotk=np.asarray(bdotk, dtype=float),
            itheta=np.asarray(itheta, dtype=float),
            izeta=np.asarray(izeta, dtype=float),
            bsubsu=np.asarray(bsubsu, dtype=float),
            bsubsv=np.asarray(bsubsv, dtype=float),
            bsubu=np.asarray(bsubu, dtype=float),
            bsubv=np.asarray(bsubv, dtype=float),
            bsubu_raw=np.asarray(bsubu_raw, dtype=float) if bsubu_raw is not None else None,
            bsubv_raw=np.asarray(bsubv_raw, dtype=float) if bsubv_raw is not None else None,
            bsupu=np.asarray(bsupu, dtype=float),
            bsupv=np.asarray(bsupv, dtype=float),
            bsubs=np.asarray(bsubs_use, dtype=float),
            bsubs_raw=np.asarray(bsubs, dtype=float),
            sqrtg=np.asarray(sqrtg, dtype=float),
            bsq=np.asarray(bsq, dtype=float),
            vp=np.asarray(vp, dtype=float),
            phips=np.asarray(phips, dtype=float),
            wint=np.asarray(wint, dtype=float),
        )

    DMerc = np.zeros((ns,), dtype=float)
    Dshear = np.zeros((ns,), dtype=float)
    Dcurr = np.zeros((ns,), dtype=float)
    Dwell = np.zeros((ns,), dtype=float)
    Dgeod = np.zeros((ns,), dtype=float)
    shear = np.zeros((ns,), dtype=float)
    vpp = np.zeros((ns,), dtype=float)
    presp = np.zeros((ns,), dtype=float)
    ip = np.zeros((ns,), dtype=float)
    torcur = np.zeros((ns,), dtype=float)
    if ns > 1:
        if exact_sum:
            for i in range(1, ns):
                torcur[i] = sign_jac * (2.0 * np.pi) * _sum_w(bsubu[i])
        else:
            torcur[1:] = sign_jac * (2.0 * np.pi) * np.einsum("sij,ij->s", bsubu[1:], wint_f, optimize=True)
    if ns > 2:
        phip_full = 0.5 * (phip_real[2:] + phip_real[1:-1])
        denom = 1.0 / (hs * phip_full)
        shear[1:-1] = (iotas[2:] - iotas[1:-1]) * denom
        vpp[1:-1] = (vp_real[2:] - vp_real[1:-1]) * denom
        presp[1:-1] = (pres[2:] - pres[1:-1]) * denom
        ip[1:-1] = (torcur[2:] - torcur[1:-1]) * denom

    b2 = 2.0 * (bsq - pres[:, None, None])
    for i in range(1, ns - 1):
        phip_full = 0.5 * (phip_real[i + 1] + phip_real[i])
        gsqrt_raw = 0.5 * (sqrtg[i] + sqrtg[i + 1])
        gsqrt_full = gsqrt_raw / phip_full
        sqs = float(np.sqrt(s[i]))
        r1f = R_even[i] + sqs * R_odd[i]
        rtf = Ru_even[i] + sqs * Ru_odd[i]
        ztf = Zu_even[i] + sqs * Zu_odd[i]
        rzf = Rv_even[i] + sqs * Rv_odd[i]
        zzf = Zv_even[i] + sqs * Zv_odd[i]
        gtt = rtf * rtf + ztf * ztf
        gpp = (gsqrt_full * gsqrt_full) / (gtt * r1f * r1f + (rtf * zzf - rzf * ztf) ** 2)
        b2i = 0.5 * (b2[i + 1] + b2[i])
        ob2 = gsqrt_full / b2i
        tpp = _sum_w(ob2)
        ob2 = b2i * gsqrt_full * gpp
        tbb = _sum_w(ob2)
        # VMEC divides bdotj by the raw Jacobian (before phip scaling),
        # then uses the flux-normalized Jacobian in jdotb.
        bdotj_norm = np.where(gsqrt_raw != 0.0, bdotk_merc[i] / gsqrt_raw, 0.0)
        jdotb = bdotj_norm * gpp * gsqrt_full
        tjb = _sum_w(jdotb)
        jdotb2 = jdotb * bdotj_norm / b2i
        tjj = _sum_w(jdotb2)

        tpp *= (2.0 * np.pi) ** 2
        tjb *= (2.0 * np.pi) ** 2
        tbb *= (2.0 * np.pi) ** 2
        tjj *= (2.0 * np.pi) ** 2
        dshear = 0.25 * shear[i] * shear[i]
        dcurr = -shear[i] * (tjb - ip[i] * tbb)
        dwell = presp[i] * (vpp[i] - presp[i] * tpp) * tbb
        dgeod = tjb * tjb - tbb * tjj
        Dshear[i] = dshear
        Dcurr[i] = dcurr
        Dwell[i] = dwell
        Dgeod[i] = dgeod
        DMerc[i] = dshear + dcurr + dwell + dgeod

    # jxbforce-style 1D diagnostics (jdotb, bdotb, bdotgradv).
    jdotb = np.zeros((ns,), dtype=float)
    bdotb = np.zeros((ns,), dtype=float)
    bdotgradv = np.zeros((ns,), dtype=float)
    dnorm1 = float((2.0 * np.pi) ** 2)
    vp = np.asarray(vp, dtype=float)
    phips = np.asarray(phips, dtype=float)
    for js in range(1, ns - 1):
        denom = vp[js + 1] + vp[js]
        if denom == 0.0:
            continue
        ovp = 2.0 / denom / dnorm1
        tjnorm = ovp * float(sign_jac)
        sqgb2 = sqrtg[js + 1] * (bsq[js + 1] - pres[js + 1]) + sqrtg[js] * (bsq[js] - pres[js])
        sigma = sigma_an[js]
        jdotb[js] = dnorm1 * tjnorm * _sum_w(bdotk[js] / sigma)
        bdotb[js] = dnorm1 * tjnorm * _sum_w(sqgb2 / sigma)
        bdotgradv[js] = 0.5 * dnorm1 * tjnorm * (phips[js] + phips[js + 1])
    if ns > 2:
        jdotb[0] = 2.0 * jdotb[1] - jdotb[2]
        jdotb[-1] = 2.0 * jdotb[-2] - jdotb[-3]
        bdotb[0] = 2.0 * bdotb[2] - bdotb[1]
        bdotb[-1] = 2.0 * bdotb[-2] - bdotb[-3]
        bdotgradv[0] = 2.0 * bdotgradv[1] - bdotgradv[2]
        bdotgradv[-1] = 2.0 * bdotgradv[-2] - bdotgradv[-3]

    return (
        np.asarray(DMerc, dtype=float),
        np.asarray(Dshear, dtype=float),
        np.asarray(Dcurr, dtype=float),
        np.asarray(Dwell, dtype=float),
        np.asarray(Dgeod, dtype=float),
        np.asarray(jdotb, dtype=float),
        np.asarray(bdotb, dtype=float),
        np.asarray(bdotgradv, dtype=float),
    )


def _apply_nyquist_half_weight(
    *,
    coeff_cos: np.ndarray,
    coeff_sin: np.ndarray,
    modes,
    trig,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply VMEC Nyquist normalization for edge modes (m=ntheta1/2, n=nzeta/2)."""
    coeff_cos = np.asarray(coeff_cos, dtype=float)
    coeff_sin = np.asarray(coeff_sin, dtype=float)
    if coeff_cos.ndim != 2 or coeff_sin.ndim != 2:
        raise ValueError("Expected coeff arrays with shape (ns, K)")

    m = np.asarray(modes.m, dtype=int)
    n = np.asarray(modes.n, dtype=int)
    if m.size == 0:
        return coeff_cos, coeff_sin

    m_nyq = int(np.max(m)) if m.size else -1
    n_nyq = int(np.max(np.abs(n))) if n.size else -1

    mask = np.zeros_like(m, dtype=bool)
    if m_nyq > 0:
        mask |= m == m_nyq
    if n_nyq > 0:
        mask |= np.abs(n) == n_nyq
    if not np.any(mask):
        return coeff_cos, coeff_sin

    coeff_cos = coeff_cos.copy()
    coeff_sin = coeff_sin.copy()
    coeff_cos[:, mask] *= 0.5
    coeff_sin[:, mask] *= 0.5
    return coeff_cos, coeff_sin


def _vmec_wrout_nyquist_cos_coeffs(
    *,
    f: np.ndarray,
    modes: ModeTable,
    trig,
) -> np.ndarray:
    """VMEC wrout-style nyquist analysis for cos coefficients (lasym=False).

    This mirrors wrout.f's tcosi integration:
      tcosi = dmult*(cosmui*cosnv + sgn*sinmui*sinnv)
      coeff = sum_{i,k} tcosi * f
    where cosmui/sinmui already include dnorm and endpoint weights.
    """
    f = np.asarray(f, dtype=float)
    if f.ndim != 3:
        raise ValueError("Expected f with shape (ns, ntheta, nzeta)")
    m = np.asarray(modes.m, dtype=int)
    n = np.asarray(modes.n, dtype=int)
    if m.size == 0:
        return np.zeros((f.shape[0], 0), dtype=float)

    nt2 = int(trig.ntheta2)
    if int(f.shape[1]) < nt2:
        raise ValueError("Input theta grid is smaller than VMEC ntheta2")
    f = f[:, :nt2, :]

    cosmui = np.asarray(trig.cosmui, dtype=float)[:nt2, :]
    sinmui = np.asarray(trig.sinmui, dtype=float)[:nt2, :]
    cosnv = np.asarray(trig.cosnv, dtype=float)
    sinnv = np.asarray(trig.sinnv, dtype=float)

    mmax = int(np.max(m))
    nmax = int(np.max(np.abs(n)))
    if cosmui.shape[1] <= mmax or cosnv.shape[1] <= nmax:
        raise ValueError("Trig tables do not cover Nyquist mode limits")

    # VMEC halves cosmui(:,mnyq) and cosnv(:,nnyq) during wrout when lnyquist.
    mnyq = cosmui.shape[1] - 1
    if mnyq > 0:
        cosmui = cosmui.copy()
        cosmui[:, mnyq] *= 0.5
    nnyq = cosnv.shape[1] - 1
    if nnyq > 0:
        cosnv = cosnv.copy()
        cosnv[:, nnyq] *= 0.5

    f_theta_cos = np.einsum("sik,im->smk", f, cosmui, optimize=True)
    f_theta_sin = np.einsum("sik,im->smk", f, sinmui, optimize=True)
    cos_zeta = np.einsum("smk,kn->smn", f_theta_cos, cosnv, optimize=True)
    sin_zeta = np.einsum("smk,kn->smn", f_theta_sin, sinnv, optimize=True)

    n_abs = np.abs(n)
    sgn = np.where(n < 0, -1.0, 1.0)
    cos_part = cos_zeta[:, m, n_abs]
    sin_part = sin_zeta[:, m, n_abs]
    coeff = cos_part + sgn[None, :] * sin_part

    mscale = np.asarray(trig.mscale, dtype=float)
    nscale = np.asarray(trig.nscale, dtype=float)
    tmult = 0.5 / float(getattr(trig, "r0scale", 1.0)) ** 2
    if int(getattr(trig, "ntheta3", nt2)) > nt2:
        # VMEC wrout doubles tmult for LASYM after switching to full-grid dnorm.
        tmult *= 2.0
    dmult = mscale[m] * nscale[n_abs] * tmult
    dmult = np.where((m == 0) | (n == 0), 2.0 * dmult, dmult)
    coeff = coeff * dmult[None, :]
    return np.asarray(coeff, dtype=float)


def _vmec_symoutput_split(
    *,
    f: np.ndarray,
    trig,
    reversed_sym: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """VMEC symoutput-style split into symmetric/antisymmetric parts on [0,π].

    Mirrors `symforce.f:symoutput`:
      sym = 0.5*(f + f_ref), asym = 0.5*(f - f_ref)
    except for `bsubs`, where the dominant symmetry is reversed:
      sym = 0.5*(f - f_ref), asym = 0.5*(f + f_ref)

    Returns arrays on the reduced theta grid (ntheta2).
    """
    f = np.asarray(f, dtype=float)
    if f.ndim != 3:
        raise ValueError("Expected f with shape (ns, ntheta, nzeta)")
    nt2 = int(trig.ntheta2)
    nt1 = int(trig.ntheta1)
    if int(f.shape[1]) < nt2:
        raise ValueError("Input theta grid is smaller than VMEC ntheta2")
    nzeta = int(f.shape[2])
    if nzeta <= 0:
        return f[:, :nt2, :].copy(), f[:, :nt2, :].copy()

    i0 = np.arange(nt2, dtype=int)
    ir0 = np.where(i0 == 0, 0, nt1 - i0)
    kk = (nzeta - np.arange(nzeta, dtype=int)) % nzeta
    f_half = f[:, :nt2, :]
    f_ref = f[:, ir0, :][:, :, kk]
    if reversed_sym:
        sym = 0.5 * (f_half - f_ref)
        asym = 0.5 * (f_half + f_ref)
    else:
        sym = 0.5 * (f_half + f_ref)
        asym = 0.5 * (f_half - f_ref)
    return sym, asym


def _vmec_symforce_apply(
    *,
    f: np.ndarray,
    trig,
    kind: str,
) -> np.ndarray:
    """Apply VMEC symforce.f to a full-grid realspace field.

    This mirrors the in-place update in symforce.f: for i<=ntheta2,
    overwrite f(i) with the symmetric piece, leaving i>ntheta2 intact.
    """
    f = np.asarray(f, dtype=float)
    if f.ndim != 3:
        raise ValueError("Expected f with shape (ns, ntheta, nzeta)")
    nt2 = int(trig.ntheta2)
    nt1 = int(trig.ntheta1)
    if int(f.shape[1]) < nt2:
        raise ValueError("Input theta grid is smaller than VMEC ntheta2")
    nzeta = int(f.shape[2])
    if nzeta <= 0:
        return f.copy()

    i0 = np.arange(nt2, dtype=int)
    ir0 = np.where(i0 == 0, 0, nt1 - i0)
    kk = (nzeta - np.arange(nzeta, dtype=int)) % nzeta
    f_half = f[:, :nt2, :]
    f_ref = f[:, ir0, :][:, :, kk]

    if kind in ("ars", "bzs", "bls", "rcs", "czs", "cls"):
        f_new = 0.5 * (f_half + f_ref)
    elif kind in ("brs", "azs", "zcs", "crs"):
        f_new = 0.5 * (f_half - f_ref)
    else:  # pragma: no cover
        raise ValueError(f"symforce: unknown kind {kind!r}")

    f_sym = np.array(f, copy=True)
    f_sym[:, :nt2, :] = f_new
    return f_sym


def _vmec_symforce_antisym(
    *,
    f: np.ndarray,
    trig,
    kind: str,
    base: np.ndarray | None = None,
) -> np.ndarray:
    """Apply VMEC symforce.f antisymmetric output (ara/bra/etc).

    The antisymmetric part is written on i<=ntheta2; values on the upper
    half remain from `base` (mirroring symforce's out-arrays).
    """
    f = np.asarray(f, dtype=float)
    if f.ndim != 3:
        raise ValueError("Expected f with shape (ns, ntheta, nzeta)")
    nt2 = int(trig.ntheta2)
    nt1 = int(trig.ntheta1)
    if int(f.shape[1]) < nt2:
        raise ValueError("Input theta grid is smaller than VMEC ntheta2")
    nzeta = int(f.shape[2])
    if nzeta <= 0:
        return np.asarray(base, dtype=float).copy() if base is not None else f.copy()

    if base is None:
        out = np.zeros_like(f)
    else:
        base = np.asarray(base, dtype=float)
        if base.shape != f.shape:
            raise ValueError("symforce antisym base shape mismatch")
        out = np.array(base, copy=True)

    i0 = np.arange(nt2, dtype=int)
    ir0 = np.where(i0 == 0, 0, nt1 - i0)
    kk = (nzeta - np.arange(nzeta, dtype=int)) % nzeta
    f_half = f[:, :nt2, :]
    f_ref = f[:, ir0, :][:, :, kk]

    if kind in ("ars", "bzs", "bls", "rcs", "czs", "cls"):
        f_new = 0.5 * (f_half - f_ref)
    elif kind in ("brs", "azs", "zcs", "crs"):
        f_new = 0.5 * (f_half + f_ref)
    else:  # pragma: no cover
        raise ValueError(f"symforce: unknown kind {kind!r}")

    out[:, :nt2, :] = f_new
    return out


def _vmec_symoutput_expand(
    *,
    sym: np.ndarray,
    asym: np.ndarray | None,
    trig,
) -> np.ndarray:
    """Expand VMEC symoutput parts back to the full [0,2π) theta grid."""
    sym = np.asarray(sym, dtype=float)
    if sym.ndim != 3:
        raise ValueError("Expected sym with shape (ns, ntheta2, nzeta)")
    if asym is None:
        asym = np.zeros_like(sym)
    else:
        asym = np.asarray(asym, dtype=float)
        if asym.shape != sym.shape:
            raise ValueError("sym/asym shape mismatch")

    ns, nt2, nzeta = sym.shape
    nt1 = int(getattr(trig, "ntheta1", nt2))
    nt3 = int(getattr(trig, "ntheta3", nt2))
    if nt3 < nt2:
        nt3 = nt2

    full = np.zeros((ns, nt3, nzeta), dtype=float)
    full[:, :nt2, :] = sym + asym
    if nt3 == nt2:
        return full

    i0 = np.arange(nt2, dtype=int)
    ir0 = np.where(i0 == 0, 0, nt1 - i0)
    kk = (nzeta - np.arange(nzeta, dtype=int)) % nzeta
    mask = ir0 >= nt2
    if np.any(mask):
        ir = ir0[mask]
        sym_ref = sym[:, mask, :][:, :, kk]
        asym_ref = asym[:, mask, :][:, :, kk]
        full[:, ir, :] = sym_ref - asym_ref
    return full


def _vmec_wrout_nyquist_sin_coeffs(
    *,
    f: np.ndarray,
    modes: ModeTable,
    trig,
) -> np.ndarray:
    """VMEC wrout-style nyquist analysis for sin coefficients (lasym=False).

    This mirrors wrout.f's tsini integration:
      tsini = dmult*(sinmui*cosnv - sgn*cosmui*sinnv)
      coeff = sum_{i,k} tsini * f
    where cosmui/sinmui already include dnorm and endpoint weights.
    """
    f = np.asarray(f, dtype=float)
    if f.ndim != 3:
        raise ValueError("Expected f with shape (ns, ntheta, nzeta)")
    m = np.asarray(modes.m, dtype=int)
    n = np.asarray(modes.n, dtype=int)
    if m.size == 0:
        return np.zeros((f.shape[0], 0), dtype=float)

    nt2 = int(trig.ntheta2)
    if int(f.shape[1]) < nt2:
        raise ValueError("Input theta grid is smaller than VMEC ntheta2")
    f = f[:, :nt2, :]

    cosmui = np.asarray(trig.cosmui, dtype=float)[:nt2, :]
    sinmui = np.asarray(trig.sinmui, dtype=float)[:nt2, :]
    cosnv = np.asarray(trig.cosnv, dtype=float)
    sinnv = np.asarray(trig.sinnv, dtype=float)

    mmax = int(np.max(m))
    nmax = int(np.max(np.abs(n)))
    if cosmui.shape[1] <= mmax or cosnv.shape[1] <= nmax:
        raise ValueError("Trig tables do not cover Nyquist mode limits")

    # VMEC halves cosmui(:,mnyq) and cosnv(:,nnyq) during wrout when lnyquist.
    mnyq = cosmui.shape[1] - 1
    if mnyq > 0:
        cosmui = cosmui.copy()
        cosmui[:, mnyq] *= 0.5
    nnyq = cosnv.shape[1] - 1
    if nnyq > 0:
        cosnv = cosnv.copy()
        cosnv[:, nnyq] *= 0.5

    f_theta_sin = np.einsum("sik,im->smk", f, sinmui, optimize=True)
    f_theta_cos = np.einsum("sik,im->smk", f, cosmui, optimize=True)
    cos_zeta = np.einsum("smk,kn->smn", f_theta_sin, cosnv, optimize=True)
    sin_zeta = np.einsum("smk,kn->smn", f_theta_cos, sinnv, optimize=True)

    n_abs = np.abs(n)
    sgn = np.where(n < 0, -1.0, 1.0)
    cos_part = cos_zeta[:, m, n_abs]
    sin_part = sin_zeta[:, m, n_abs]
    coeff = cos_part - sgn[None, :] * sin_part

    mscale = np.asarray(trig.mscale, dtype=float)
    nscale = np.asarray(trig.nscale, dtype=float)
    tmult = 0.5 / float(getattr(trig, "r0scale", 1.0)) ** 2
    if int(getattr(trig, "ntheta3", nt2)) > nt2:
        # VMEC wrout doubles tmult for LASYM after switching to full-grid dnorm.
        tmult *= 2.0
    dmult = mscale[m] * nscale[n_abs] * tmult
    dmult = np.where((m == 0) | (n == 0), 2.0 * dmult, dmult)
    coeff = coeff * dmult[None, :]
    return np.asarray(coeff, dtype=float)


def _vmec_wrout_nyquist_lasym_loop(
    *,
    bsq: np.ndarray,
    gsqrt: np.ndarray,
    bsubu: np.ndarray,
    bsubv: np.ndarray,
    bsubs: np.ndarray,
    bsupu: np.ndarray,
    bsupv: np.ndarray,
    modes: ModeTable,
    trig,
) -> dict[str, np.ndarray]:
    """VMEC wrout.f lasym loop for Nyquist coefficients (parity-accurate).

    Mirrors the RADIUS2/RADIUS3 loops in wrout.f:
    - Use symoutput-split fields on the reduced theta interval.
    - Compute cosine/sine coefficients via explicit (j,k) summations.
    """
    bsq = np.asarray(bsq, dtype=float)
    gsqrt = np.asarray(gsqrt, dtype=float)
    bsubu = np.asarray(bsubu, dtype=float)
    bsubv = np.asarray(bsubv, dtype=float)
    bsubs = np.asarray(bsubs, dtype=float)
    bsupu = np.asarray(bsupu, dtype=float)
    bsupv = np.asarray(bsupv, dtype=float)

    if bsq.ndim != 3:
        raise ValueError("Expected bsq with shape (ns, ntheta, nzeta)")
    ns, nt2, nzeta = bsq.shape
    if int(trig.ntheta2) != nt2:
        raise ValueError("lasym wrout expects reduced theta grid (ntheta2)")

    m = np.asarray(modes.m, dtype=int)
    n = np.asarray(modes.n, dtype=int)
    if m.size == 0:
        z = np.zeros((ns, 0), dtype=float)
        return {
            "gmnc": z.copy(),
            "bmnc": z.copy(),
            "bsubumnc": z.copy(),
            "bsubvmnc": z.copy(),
            "bsubsmns": z.copy(),
            "bsupumnc": z.copy(),
            "bsupvmnc": z.copy(),
            "gmns": z.copy(),
            "bmns": z.copy(),
            "bsubumns": z.copy(),
            "bsubvmns": z.copy(),
            "bsubsmnc": z.copy(),
            "bsupumns": z.copy(),
            "bsupvmns": z.copy(),
        }

    cosmui = np.asarray(trig.cosmui, dtype=float)[:nt2, :]
    sinmui = np.asarray(trig.sinmui, dtype=float)[:nt2, :]
    cosnv = np.asarray(trig.cosnv, dtype=float)
    sinnv = np.asarray(trig.sinnv, dtype=float)
    mscale = np.asarray(trig.mscale, dtype=float)
    nscale = np.asarray(trig.nscale, dtype=float)
    tmult = 0.5 / float(getattr(trig, "r0scale", 1.0)) ** 2
    if int(getattr(trig, "ntheta3", nt2)) > nt2:
        # VMEC wrout doubles tmult for LASYM after switching to full-grid dnorm.
        tmult *= 2.0

    mnyq = cosmui.shape[1] - 1
    if mnyq > 0:
        cosmui = cosmui.copy()
        cosmui[:, mnyq] *= 0.5
    nnyq = cosnv.shape[1] - 1
    if nnyq > 0:
        cosnv = cosnv.copy()
        cosnv[:, nnyq] *= 0.5

    # Output arrays (ns, K).
    K = int(m.size)
    gmnc = np.zeros((ns, K), dtype=float)
    bmnc = np.zeros((ns, K), dtype=float)
    bsubumnc = np.zeros((ns, K), dtype=float)
    bsubvmnc = np.zeros((ns, K), dtype=float)
    bsubsmns = np.zeros((ns, K), dtype=float)
    bsupumnc = np.zeros((ns, K), dtype=float)
    bsupvmnc = np.zeros((ns, K), dtype=float)
    gmns = np.zeros((ns, K), dtype=float)
    bmns = np.zeros((ns, K), dtype=float)
    bsubumns = np.zeros((ns, K), dtype=float)
    bsubvmns = np.zeros((ns, K), dtype=float)
    bsubsmnc = np.zeros((ns, K), dtype=float)
    bsupumns = np.zeros((ns, K), dtype=float)
    bsupvmns = np.zeros((ns, K), dtype=float)

    # Loop over radial surfaces (skip axis; wrout uses js=2..ns).
    for js in range(1, ns):
        bsq_s = bsq[js]
        gsqrt_s = gsqrt[js]
        bsubu_s = bsubu[js]
        bsubv_s = bsubv[js]
        bsubs_s = bsubs[js]
        bsupu_s = bsupu[js]
        bsupv_s = bsupv[js]
        for mn in range(K):
            mval = int(m[mn])
            nval = int(n[mn])
            n1 = abs(nval)
            dmult = mscale[mval] * nscale[n1] * tmult
            if mval == 0 or nval == 0:
                dmult *= 2.0
            sgn = 1.0 if nval >= 0 else -1.0

            # symmetric (cos) channel coefficients
            gmn = 0.0
            bmn = 0.0
            bsubumn = 0.0
            bsubvmn = 0.0
            bsubsmn = 0.0
            bsupumn = 0.0
            bsupvmn = 0.0

            # asymmetric (sin) channel coefficients
            gmn_a = 0.0
            bmn_a = 0.0
            bsubumn_a = 0.0
            bsubvmn_a = 0.0
            bsubsmn_a = 0.0
            bsupumn_a = 0.0
            bsupvmn_a = 0.0

            for j in range(nt2):
                cosmu_j = cosmui[j, mval]
                sinmu_j = sinmui[j, mval]
                for k in range(nzeta):
                    tcosi = dmult * (cosmu_j * cosnv[k, n1] + sgn * sinmu_j * sinnv[k, n1])
                    tsini = dmult * (sinmu_j * cosnv[k, n1] - sgn * cosmu_j * sinnv[k, n1])
                    # symmetric channel
                    gmn += tcosi * gsqrt_s[j, k]
                    bmn += tcosi * bsq_s[j, k]
                    bsubumn += tcosi * bsubu_s[j, k]
                    bsubvmn += tcosi * bsubv_s[j, k]
                    bsubsmn += tsini * bsubs_s[j, k]
                    bsupumn += tcosi * bsupu_s[j, k]
                    bsupvmn += tcosi * bsupv_s[j, k]
                    # asymmetric channel
                    gmn_a += tsini * gsqrt_s[j, k]
                    bmn_a += tsini * bsq_s[j, k]
                    bsubumn_a += tsini * bsubu_s[j, k]
                    bsubvmn_a += tsini * bsubv_s[j, k]
                    bsubsmn_a += tcosi * bsubs_s[j, k]
                    bsupumn_a += tsini * bsupu_s[j, k]
                    bsupvmn_a += tsini * bsupv_s[j, k]

            gmnc[js, mn] = gmn
            bmnc[js, mn] = bmn
            bsubumnc[js, mn] = bsubumn
            bsubvmnc[js, mn] = bsubvmn
            bsubsmns[js, mn] = bsubsmn
            bsupumnc[js, mn] = bsupumn
            bsupvmnc[js, mn] = bsupvmn

            gmns[js, mn] = gmn_a
            bmns[js, mn] = bmn_a
            bsubumns[js, mn] = bsubumn_a
            bsubvmns[js, mn] = bsubvmn_a
            bsubsmnc[js, mn] = bsubsmn_a
            bsupumns[js, mn] = bsupumn_a
            bsupvmns[js, mn] = bsupvmn_a

    # Axis values and bsubs extrapolation (wrout.f).
    if ns > 0:
        gmnc[0, :] = 0.0
        bmnc[0, :] = 0.0
        bsubumnc[0, :] = 0.0
        bsubvmnc[0, :] = 0.0
        bsupumnc[0, :] = 0.0
        bsupvmnc[0, :] = 0.0
        gmns[0, :] = 0.0
        bmns[0, :] = 0.0
        bsubumns[0, :] = 0.0
        bsubvmns[0, :] = 0.0
        bsupumns[0, :] = 0.0
        bsupvmns[0, :] = 0.0
    if ns > 2:
        bsubsmns[0, :] = 2.0 * bsubsmns[1, :] - bsubsmns[2, :]
        bsubsmnc[0, :] = 2.0 * bsubsmnc[1, :] - bsubsmnc[2, :]

    return dict(
        gmnc=gmnc,
        bmnc=bmnc,
        bsubumnc=bsubumnc,
        bsubvmnc=bsubvmnc,
        bsubsmns=bsubsmns,
        bsupumnc=bsupumnc,
        bsupvmnc=bsupvmnc,
        gmns=gmns,
        bmns=bmns,
        bsubumns=bsubumns,
        bsubvmns=bsubvmns,
        bsubsmnc=bsubsmnc,
        bsupumns=bsupumns,
        bsupvmns=bsupvmns,
    )


def _vmec_wrout_lasym_bsubuv_output_scale(
    *,
    bsubumnc: np.ndarray,
    bsubvmnc: np.ndarray,
    bsubumns: np.ndarray,
    bsubvmns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply VMEC's LASYM `wrout` scaling for covariant bsubu/bsubv coefficients."""

    return (
        2.0 * np.asarray(bsubumnc, dtype=float),
        2.0 * np.asarray(bsubvmnc, dtype=float),
        2.0 * np.asarray(bsubumns, dtype=float),
        2.0 * np.asarray(bsubvmns, dtype=float),
    )


def _vmec_wrout_nyquist_synthesis(
    *,
    coeff_c: np.ndarray,
    coeff_s: np.ndarray,
    modes: ModeTable,
    trig,
) -> np.ndarray:
    """Synthesize real-space field from wrout-style Nyquist coefficients (lasym=False)."""
    coeff_c = np.asarray(coeff_c, dtype=float)
    coeff_s = np.asarray(coeff_s, dtype=float)
    if coeff_c.ndim != 2 or coeff_s.ndim != 2:
        raise ValueError("Expected coeff arrays with shape (ns, K)")
    m = np.asarray(modes.m, dtype=int)
    n = np.asarray(modes.n, dtype=int)
    if m.size == 0:
        return np.zeros((coeff_c.shape[0], 0, 0), dtype=float)

    nt2 = int(trig.ntheta2)
    cosmu = np.asarray(trig.cosmu, dtype=float)[:nt2, :]
    sinmu = np.asarray(trig.sinmu, dtype=float)[:nt2, :]
    cosnv = np.asarray(trig.cosnv, dtype=float)
    sinnv = np.asarray(trig.sinnv, dtype=float)

    mmax = int(np.max(m))
    nmax = int(np.max(np.abs(n)))
    if cosmu.shape[1] <= mmax or cosnv.shape[1] <= nmax:
        raise ValueError("Trig tables do not cover Nyquist mode limits")

    n_abs = np.abs(n)
    sgn = np.where(n < 0, -1.0, 1.0)

    mscale = np.asarray(trig.mscale, dtype=float)
    nscale = np.asarray(trig.nscale, dtype=float)
    tmult = 0.5 / float(getattr(trig, "r0scale", 1.0)) ** 2
    dmult = mscale[m] * nscale[n_abs] * tmult
    dmult = np.where((m == 0) | (n == 0), 2.0 * dmult, dmult)
    dmult = np.where(dmult == 0.0, 1.0, dmult)

    raw_c = coeff_c / dmult[None, :]
    raw_s = coeff_s / dmult[None, :]

    cosmu_m = cosmu[:, m]
    sinmu_m = sinmu[:, m]
    cosnv_n = cosnv[:, n_abs]
    sinnv_n = sinnv[:, n_abs] * sgn[None, :]

    term_c = cosmu_m[:, None, :] * cosnv_n[None, :, :] + sinmu_m[:, None, :] * sinnv_n[None, :, :]
    term_s = sinmu_m[:, None, :] * cosnv_n[None, :, :] - cosmu_m[:, None, :] * sinnv_n[None, :, :]

    f = np.einsum("sk,ijk->sij", raw_c, term_c, optimize=True) + np.einsum("sk,ijk->sij", raw_s, term_s, optimize=True)
    return np.asarray(f, dtype=float)


def _vmec_wrout_nyquist_sin_coeffs_loop(
    *,
    f: np.ndarray,
    modes: ModeTable,
    trig,
) -> np.ndarray:
    """Loop-based wrout nyquist sin coefficients (matches VMEC summation order)."""
    f = np.asarray(f, dtype=float)
    if f.ndim != 3:
        raise ValueError("Expected f with shape (ns, ntheta, nzeta)")
    m_arr = np.asarray(modes.m, dtype=int)
    n_arr = np.asarray(modes.n, dtype=int)
    ns = int(f.shape[0])
    if m_arr.size == 0:
        return np.zeros((ns, 0), dtype=float)

    nt2 = int(trig.ntheta2)
    if int(f.shape[1]) < nt2:
        raise ValueError("Input theta grid is smaller than VMEC ntheta2")
    f = f[:, :nt2, :]

    cosmui = np.asarray(trig.cosmui, dtype=float)[:nt2, :]
    sinmui = np.asarray(trig.sinmui, dtype=float)[:nt2, :]
    cosnv = np.asarray(trig.cosnv, dtype=float)
    sinnv = np.asarray(trig.sinnv, dtype=float)

    mmax = int(np.max(m_arr))
    nmax = int(np.max(np.abs(n_arr)))
    if cosmui.shape[1] <= mmax or cosnv.shape[1] <= nmax:
        raise ValueError("Trig tables do not cover Nyquist mode limits")

    # wrout.f applies Nyquist half-weighting by scaling transform tables
    # before the mode loops when `lnyquist` is enabled.
    mnyq = cosmui.shape[1] - 1
    if mnyq > 0:
        cosmui = cosmui.copy()
        cosmui[:, mnyq] *= 0.5
    nnyq = cosnv.shape[1] - 1
    if nnyq > 0:
        cosnv = cosnv.copy()
        cosnv[:, nnyq] *= 0.5

    mscale = np.asarray(trig.mscale, dtype=float)
    nscale = np.asarray(trig.nscale, dtype=float)
    tmult = 0.5 / float(getattr(trig, "r0scale", 1.0)) ** 2

    coeff = np.zeros((ns, m_arr.size), dtype=float)
    nzeta = int(f.shape[2])
    for js in range(ns):
        f_js = f[js]
        for idx, (m, n) in enumerate(zip(m_arr, n_arr)):
            n1 = abs(int(n))
            sgn = -1.0 if n < 0 else 1.0
            dmult = mscale[m] * nscale[n1] * tmult
            if m == 0 or n == 0:
                dmult *= 2.0
            acc = 0.0
            for k in range(nzeta):
                for j in range(nt2):
                    tsini = dmult * (sinmui[j, m] * cosnv[k, n1] - sgn * cosmui[j, m] * sinnv[k, n1])
                    acc += tsini * f_js[j, k]
            coeff[js, idx] = acc

    return coeff


def _vmec_jxbforce_cos_coeffs(
    *,
    f: np.ndarray,
    modes: ModeTable,
    trig,
) -> np.ndarray:
    """VMEC jxbforce-style cosine coefficients (lasym=False).

    This matches the low-pass filter in jxbforce.f:
      tcosi = dnorm1*(cosmui*cosnv + sgn*sinmui*sinnv)
      coeff = sum_{i,k} tcosi * f
    with dnorm1 = 1/r0scale**2 and Nyquist half-weights for m==mnyq or n==nnyq.
    """
    f = np.asarray(f, dtype=float)
    if f.ndim != 3:
        raise ValueError("Expected f with shape (ns, ntheta, nzeta)")
    m = np.asarray(modes.m, dtype=int)
    n = np.asarray(modes.n, dtype=int)
    if m.size == 0:
        return np.zeros((f.shape[0], 0), dtype=float)

    nt2 = int(trig.ntheta2)
    if int(f.shape[1]) < nt2:
        raise ValueError("Input theta grid is smaller than VMEC ntheta2")
    f = f[:, :nt2, :]

    cosmui = np.asarray(trig.cosmui, dtype=float)[:nt2, :]
    sinmui = np.asarray(trig.sinmui, dtype=float)[:nt2, :]
    cosnv = np.asarray(trig.cosnv, dtype=float)
    sinnv = np.asarray(trig.sinnv, dtype=float)

    mmax = int(np.max(m))
    nmax = int(np.max(np.abs(n)))
    if cosmui.shape[1] <= mmax or cosnv.shape[1] <= nmax:
        raise ValueError("Trig tables do not cover Nyquist mode limits")

    f_theta_cos = np.einsum("sik,im->smk", f, cosmui, optimize=True)
    f_theta_sin = np.einsum("sik,im->smk", f, sinmui, optimize=True)
    cos_zeta = np.einsum("smk,kn->smn", f_theta_cos, cosnv, optimize=True)
    sin_zeta = np.einsum("smk,kn->smn", f_theta_sin, sinnv, optimize=True)

    n_abs = np.abs(n)
    sgn = np.where(n < 0, -1.0, 1.0)
    cos_part = cos_zeta[:, m, n_abs]
    sin_part = sin_zeta[:, m, n_abs]
    coeff = cos_part + sgn[None, :] * sin_part

    r0scale = float(getattr(trig, "r0scale", 1.0))
    dmult = np.full_like(m, 1.0 / (r0scale**2), dtype=float)
    mnyq = cosmui.shape[1] - 1
    nnyq = cosnv.shape[1] - 1
    if mnyq > 0:
        dmult = np.where(m == mnyq, 0.5 * dmult, dmult)
    if nnyq > 0:
        dmult = np.where((n_abs == nnyq) & (n_abs != 0), 0.5 * dmult, dmult)
    coeff = coeff * dmult[None, :]
    return np.asarray(coeff, dtype=float)


def _vmec_jxbforce_sin_coeffs(
    *,
    f: np.ndarray,
    modes: ModeTable,
    trig,
) -> np.ndarray:
    """VMEC jxbforce-style sine coefficients (lasym=False)."""
    f = np.asarray(f, dtype=float)
    if f.ndim != 3:
        raise ValueError("Expected f with shape (ns, ntheta, nzeta)")
    m = np.asarray(modes.m, dtype=int)
    n = np.asarray(modes.n, dtype=int)
    if m.size == 0:
        return np.zeros((f.shape[0], 0), dtype=float)

    nt2 = int(trig.ntheta2)
    if int(f.shape[1]) < nt2:
        raise ValueError("Input theta grid is smaller than VMEC ntheta2")
    f = f[:, :nt2, :]

    cosmui = np.asarray(trig.cosmui, dtype=float)[:nt2, :]
    sinmui = np.asarray(trig.sinmui, dtype=float)[:nt2, :]
    cosnv = np.asarray(trig.cosnv, dtype=float)
    sinnv = np.asarray(trig.sinnv, dtype=float)

    mmax = int(np.max(m))
    nmax = int(np.max(np.abs(n)))
    if cosmui.shape[1] <= mmax or cosnv.shape[1] <= nmax:
        raise ValueError("Trig tables do not cover Nyquist mode limits")

    f_theta_sin = np.einsum("sik,im->smk", f, sinmui, optimize=True)
    f_theta_cos = np.einsum("sik,im->smk", f, cosmui, optimize=True)
    cos_zeta = np.einsum("smk,kn->smn", f_theta_sin, cosnv, optimize=True)
    sin_zeta = np.einsum("smk,kn->smn", f_theta_cos, sinnv, optimize=True)

    n_abs = np.abs(n)
    sgn = np.where(n < 0, -1.0, 1.0)
    cos_part = cos_zeta[:, m, n_abs]
    sin_part = sin_zeta[:, m, n_abs]
    coeff = cos_part - sgn[None, :] * sin_part

    r0scale = float(getattr(trig, "r0scale", 1.0))
    dmult = np.full_like(m, 1.0 / (r0scale**2), dtype=float)
    mnyq = cosmui.shape[1] - 1
    nnyq = cosnv.shape[1] - 1
    if mnyq > 0:
        dmult = np.where(m == mnyq, 0.5 * dmult, dmult)
    if nnyq > 0:
        dmult = np.where((n_abs == nnyq) & (n_abs != 0), 0.5 * dmult, dmult)
    coeff = coeff * dmult[None, :]
    return np.asarray(coeff, dtype=float)


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
        write_float_variable(ds, "D_R", ("radius",), np.asarray(getattr(wout, "D_R", np.zeros((ns,), dtype=float))))
        write_float_variable(ds, "HGlasser", ("radius",), np.asarray(getattr(wout, "H", np.zeros((ns,), dtype=float))))
        write_float_variable(
            ds,
            "GlasserCorrection",
            ("radius",),
            np.asarray(getattr(wout, "glasser_correction", np.zeros((ns,), dtype=float))),
        )
        write_float_variable(
            ds,
            "GlasserShearValid",
            ("radius",),
            np.asarray(getattr(wout, "glasser_shear_valid", np.zeros((ns,), dtype=bool)), dtype=float),
        )

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

    # Current profile metadata for vmecPlot2.
    pcurr_type = indata.get("PCURR_TYPE", None)
    if pcurr_type is None:
        pcurr_type = "power_series"
    pcurr_type = str(pcurr_type)

    piota_type = indata.get("PIOTA_TYPE", None)
    if piota_type is None:
        piota_type = "power_series"
    piota_type = str(piota_type)

    ac_raw = indata.get("AC", [])
    if isinstance(ac_raw, (int, float, np.floating)):
        ac_vals = [float(ac_raw)]
    elif isinstance(ac_raw, list):
        ac_vals = [float(v) for v in ac_raw]
    else:
        ac_vals = []
    n_preset = max(21, len(ac_vals) if ac_vals else 1)
    ac = np.zeros((n_preset,), dtype=float)
    for i, v in enumerate(ac_vals):
        if i >= n_preset:
            break
        ac[i] = v

    ndfmax = 101
    ac_aux_s = -np.ones((ndfmax,), dtype=float)
    ac_aux_f = np.zeros((ndfmax,), dtype=float)

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
