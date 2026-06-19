"""Minimal `wout_*.nc` reader helpers.

This module is intentionally small and only depends on `netCDF4` when used.
It is meant for regression comparisons against VMEC2000 outputs.
"""

from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from ._compat import has_jax, jax, jnp
from .modes import vmec_mode_table
from .modes import nyquist_mode_table_from_grid
from .state import VMECState
from .fourier import eval_fourier
from .vmec_parity import vmec_m1_internal_to_physical_signed
from .vmec_realspace import (
    vmec_realspace_synthesis,
    vmec_realspace_synthesis_dtheta,
    vmec_realspace_geom_from_state,
)
from .vmec_residue import vmec_pwint_from_trig, vmec_wint_from_trig
from .io.wout import bsubs as _wout_bsubs_helpers
from .io.wout import debug as _wout_debug_helpers
from .io.wout import diagnostics as _wout_diagnostics
from .io.wout import flux as _wout_flux_helpers
from .io.wout import jxbforce as _wout_jxbforce_helpers
from .io.wout import mercier as _wout_mercier
from .io.wout import parity as _wout_parity_helpers
from .io.wout import state as _wout_state_helpers
from .io.wout.minimal import (
    WoutMinimalVmecLike,
    attach_force_payload_geometry,
    build_main_geometry_coefficients,
    build_minimal_wout_data_kwargs,
    compute_minimal_wout_scalar_diagnostics,
    device_get_if_available,
    env_enabled,
    indata_for_wout_force_path,
    lbsubs_from_indata_and_env,
    minimal_wout_field_options_from_env,
    minimal_wout_runtime_options_from_env,
    pressure_profiles_from_mass_vp,
    prepare_wout_bss_source_payload,
    prepare_wout_bcovar_payload,
    prepare_profile_payload,
)
from .io.wout.netcdf import (
    read_wout_payload,
    read_wout_scalar_metadata,
    write_wout_payload,
)
from .io.wout.nyquist import (
    apply_nyquist_half_weight as _apply_nyquist_half_weight,  # noqa: F401 - compatibility export
    minimal_wout_symmetric_nyquist_coefficients,
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
    vmec_wrout_nyquist_sin_coeffs_loop as _vmec_wrout_nyquist_sin_coeffs_loop,  # noqa: F401 - compatibility export
    vmec_wrout_nyquist_synthesis as _vmec_wrout_nyquist_synthesis,  # noqa: F401 - compatibility export
)
from .io.wout.schema import (
    WoutData,
    _bool_from_nc,  # noqa: F401 - compatibility export
    _nc_scalar,  # noqa: F401 - compatibility export
    assert_main_modes_match_wout,
)
from .vmec_tomnsp import vmec_trig_tables


MU0 = 4e-7 * np.pi  # N/A^2
_chipf_from_chips = _wout_flux_helpers.chipf_from_chips
_compute_aspectratio = _wout_diagnostics.compute_aspectratio
_compute_ctor_from_buco = _wout_diagnostics.compute_ctor_from_buco
_compute_eqfor_beta = _wout_diagnostics.compute_eqfor_beta
_compute_eqfor_betaxis = _wout_diagnostics.compute_eqfor_betaxis
_compute_equif_wout = _wout_diagnostics.compute_equif_wout
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


def _solve_jxbforce_collocation(A: np.ndarray, rhs: np.ndarray) -> np.ndarray | None:
    try:
        return np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        try:
            sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
            return sol
        except np.linalg.LinAlgError:
            return None


def _timing_start(enabled: bool) -> float | None:
    return time.perf_counter() if bool(enabled) else None


def _record_timing(timing: dict[str, float], key: str, start: float | None) -> None:
    if start is not None:
        timing[key] = time.perf_counter() - start


def _validate_wint_trig(trig) -> int:
    cosmui3 = np.asarray(trig.cosmui3)
    mscale = np.asarray(trig.mscale)
    if cosmui3.ndim != 2:
        raise ValueError("Expected trig.cosmui3 with shape (ntheta3, mmax+1)")
    if mscale.size == 0:
        raise ValueError("Expected non-empty trig.mscale")
    return int(np.asarray(trig.cosnv).shape[0])


def _vmec_wint_from_trig(trig) -> np.ndarray:
    """Return VMEC-style angular weights (wint) on the internal grid."""
    nzeta = _validate_wint_trig(trig)
    if hasattr(trig, "ntheta3"):
        return np.asarray(vmec_wint_from_trig(trig, nzeta=nzeta), dtype=float)
    w_theta = np.asarray(trig.cosmui3)[:, 0] / float(np.asarray(trig.mscale)[0])
    return np.asarray(w_theta[:, None] * np.ones((nzeta,), dtype=w_theta.dtype)[None, :], dtype=float)


def _vmec_wint_from_trig_jax(trig):
    """Return VMEC-style angular weights on the internal grid as JAX arrays."""
    nzeta = _validate_wint_trig(trig)
    if hasattr(trig, "ntheta3"):
        return vmec_wint_from_trig(trig, nzeta=nzeta)
    w_theta = jnp.asarray(trig.cosmui3)[:, 0] / jnp.asarray(trig.mscale)[0]
    return w_theta[:, None] * jnp.ones((nzeta,), dtype=w_theta.dtype)[None, :]


_bcovar_from_force_payload_with_geometry = attach_force_payload_geometry
_device_get_if_available = device_get_if_available
_env_enabled = env_enabled
_indata_for_wout_force_path = indata_for_wout_force_path
_prepare_wout_bcovar_payload = prepare_wout_bcovar_payload


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


def _synthesize_wout_geometry_from_state(
    *,
    state: VMECState,
    static,
    trig,
    light: bool,
    timing_enabled: bool,
    timing: dict[str, float],
) -> dict[str, np.ndarray | None]:
    """Synthesize real-space geometry for minimal WOUT construction."""
    from .vmec_numpy_forces import _numpy_module_patch

    if timing_enabled:
        import time as _time

        t0 = _time.perf_counter()
    with _numpy_module_patch():
        if light:
            geom = _vmec_realspace_geom_light_from_state(state=state, modes=static.modes, trig=trig)
        else:
            geom = vmec_realspace_geom_from_state(state=state, modes=static.modes, trig=trig)
    if has_jax():
        try:
            geom = jax.device_get(geom)
        except Exception:
            pass
    geom = {k: (None if v is None else np.asarray(v)) for k, v in geom.items()}
    if timing_enabled:
        timing["geom_synthesis_s"] = _time.perf_counter() - t0
    return geom


def _apply_bsubv_equif_correction(*, bsubv: np.ndarray, bsubv_e: np.ndarray, trig) -> np.ndarray:
    return _wout_diagnostics.apply_bsubv_equif_correction(
        bsubv=bsubv,
        bsubv_e=bsubv_e,
        trig=trig,
        vmec_pwint_from_trig_func=vmec_pwint_from_trig,
    )


_compute_bsubs_half_mesh = _wout_bsubs_helpers.compute_bsubs_half_mesh
_bsubuv_parity_from_state = _wout_bsubs_helpers.bsubuv_parity_from_state
_bsubuv_parity_from_coeffs = _wout_bsubs_helpers.bsubuv_parity_from_coeffs
_bsubuv_parity_from_realspace_jxbforce = _wout_bsubs_helpers.bsubuv_parity_from_realspace_jxbforce
_bsubuv_parity_from_bcovar = _wout_bsubs_helpers.bsubuv_parity_from_bcovar


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

    sol = _solve_jxbforce_collocation(A, rhs)
    if sol is None:
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

    sol = _solve_jxbforce_collocation(A, rhs)
    if sol is None:
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


def _jxbforce_float_arrays(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    return tuple(np.asarray(arr, dtype=float) for arr in arrays)


def _jxbforce_trig_slices(trig, *, nt2: int, mnyq: int, nnyq: int) -> tuple[np.ndarray, ...]:
    return (
        np.asarray(trig.cosmu, dtype=float)[:nt2, : mnyq + 1],
        np.asarray(trig.sinmu, dtype=float)[:nt2, : mnyq + 1],
        np.asarray(trig.cosmum, dtype=float)[:nt2, : mnyq + 1],
        np.asarray(trig.sinmum, dtype=float)[:nt2, : mnyq + 1],
        np.asarray(trig.cosnv, dtype=float)[:, : nnyq + 1],
        np.asarray(trig.sinnv, dtype=float)[:, : nnyq + 1],
        np.asarray(trig.cosnvn, dtype=float)[:, : nnyq + 1],
        np.asarray(trig.sinnvn, dtype=float)[:, : nnyq + 1],
    )


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
    bsubu, bsubv, bsubs, bsubsu, bsubsv, bsupu, bsupv, sqrtg, pres, vp = _jxbforce_float_arrays(
        bsubu, bsubv, bsubs, bsubsu, bsubsv, bsupu, bsupv, sqrtg, pres, vp
    )

    ns, nt2, nzeta = bsubu.shape
    if ns < 3 or hs == 0.0:
        return bsubs, bsubsu, bsubsv

    ohs = 1.0 / hs
    mnyq, nnyq = _jxbforce_nyquist_limits(trig)
    cosmu, sinmu, cosmum, sinmum, cosnv, sinnv, cosnvn, sinnvn = _jxbforce_trig_slices(
        trig, nt2=nt2, mnyq=mnyq, nnyq=nnyq
    )

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
    bsubu, bsubv, bsubs, bsubsu, bsubsv, bsupu, bsupv, sqrtg, pres, vp = _jxbforce_float_arrays(
        bsubu, bsubv, bsubs, bsubsu, bsubsv, bsupu, bsupv, sqrtg, pres, vp
    )

    ns, nt2, nzeta = bsubu.shape
    nt1 = int(getattr(trig, "ntheta1", nt2))
    nt3 = int(getattr(trig, "ntheta3", nt2))
    if nt3 < nt2:
        nt3 = nt2
    if ns < 3 or hs == 0.0:
        return bsubs, bsubsu, bsubsv

    ohs = 1.0 / hs
    mnyq, nnyq = _jxbforce_nyquist_limits(trig)
    cosmu, sinmu, cosmum, sinmum, cosnv, sinnv, cosnvn, sinnvn = _jxbforce_trig_slices(
        trig, nt2=nt2, mnyq=mnyq, nnyq=nnyq
    )

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

    return WoutData(
        **read_wout_payload(
            path,
            mu0=MU0,
            phi_profile_from_variables_func=_wout_phi_profile_from_variables,
            glasser_profiles_from_variables_func=_glasser_profiles_from_wout_variables,
        )
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

    write_wout_payload(
        path,
        wout,
        overwrite=overwrite,
        mu0=MU0,
        glasser_profiles_from_wout_data_func=_glasser_profiles_from_wout_data,
        getenv=os.getenv,
        print_func=print,
    )


def _zero_first_surface(*arrays: np.ndarray) -> None:
    """Apply VMEC wrout's zero-axis convention to coefficient arrays in-place."""

    for arr in arrays:
        if arr.shape[0] > 0:
            arr[0, :] = 0.0


def _force_sym_for_wout(arr, *, trig, lasym: bool, kind: str) -> np.ndarray:
    arr_np = np.asarray(arr, dtype=float)
    if bool(lasym) or int(arr_np.shape[1]) < int(trig.ntheta1):
        return arr_np
    return _vmec_symforce_apply(f=arr_np, trig=trig, kind=kind)


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

    runtime_options = minimal_wout_runtime_options_from_env()
    wout_timing_enabled = runtime_options.timing_enabled
    wout_light = runtime_options.light
    wout_fast_bcovar = runtime_options.fast_bcovar
    field_options = minimal_wout_field_options_from_env(wout_light=bool(wout_light))
    wout_timing: dict[str, float] = {}
    t_wout_total_start = _timing_start(bool(wout_timing_enabled))

    if converged is None:
        converged = True

    lbsubs = lbsubs_from_indata_and_env(indata)

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
    t0 = _timing_start(bool(wout_timing_enabled))
    trig = vmec_trig_tables(
        ntheta=int(cfg.ntheta),
        nzeta=int(cfg.nzeta),
        nfp=int(nfp),
        mmax=int(mmax_base),
        nmax=int(nmax_base),
        lasym=bool(lasym),
        dtype=np.asarray(state.Rcos).dtype,
    )
    _record_timing(wout_timing, "trig_tables_s", t0)

    geom = _synthesize_wout_geometry_from_state(
        state=state,
        static=static,
        trig=trig,
        light=bool(wout_light),
        timing_enabled=bool(wout_timing_enabled),
        timing=wout_timing,
    )

    # Flux and profiles on VMEC half mesh.
    s = np.asarray(static.s)
    profile_payload = prepare_profile_payload(
        state=state,
        static=static,
        indata=indata,
        modes=main_modes,
        s=s,
        ns=int(ns),
        signgs=int(signgs),
        flux_override=flux_override,
        profiles_override=profiles_override,
        equilibrium_iota_profiles_from_state_func=equilibrium_iota_profiles_from_state,
        chipf_from_chips_func=_chipf_from_chips,
    )
    (flux, chipf_wout, _, pres, _, mass, ncurr, iotas, iotaf, gamma, phipf_internal) = profile_payload

    lconm1 = bool(getattr(cfg, "lconm1", True))
    main_geom = build_main_geometry_coefficients(
        state=state,
        modes=main_modes,
        ntor=int(ntor),
        lasym=bool(lasym),
        lconm1=bool(lconm1),
    )

    # Toroidal flux (VMEC `phi`) in physical units.
    phipf_out = phipf_internal * float(2.0 * np.pi * signgs)
    chipf_out = np.asarray(chipf_wout, dtype=float) * float(2.0 * np.pi * signgs)
    phi = np.asarray(cumrect_s_halfmesh(phipf_out, s))

    # Build VMEC parity grids for Nyquist outputs.
    wout_like = WoutMinimalVmecLike(
        flux=flux,
        chipf=np.asarray(chipf_wout),
        iotaf=np.asarray(iotaf),
        iotas=np.asarray(iotas),
        signgs=int(signgs),
        nfp=int(nfp),
        mpol=int(mpol),
        ntor=int(ntor),
        lasym=bool(lasym),
        ncurr=int(ncurr),
        mass=np.asarray(mass),
        gamma=float(gamma),
        indata=indata,
        s_full=np.asarray(s, dtype=float),
        icurv_full_mesh_from_indata_func=_icurv_full_mesh_from_indata,
    )
    from .vmec_forces import vmec_forces_rz_from_wout
    from .vmec_numpy_forces import _numpy_module_patch

    bcovar_payload = _prepare_wout_bcovar_payload(
        state=state,
        static=static,
        indata=indata,
        wout_like=wout_like,
        pres=pres,
        geom=geom,
        force_payload_override=force_payload_override,
        fast_bcovar=bool(wout_fast_bcovar),
        timing_enabled=bool(wout_timing_enabled),
        timing=wout_timing,
        vmec_bcovar_half_mesh_from_wout_func=vmec_bcovar_half_mesh_from_wout,
        vmec_forces_rz_from_wout_func=vmec_forces_rz_from_wout,
        numpy_module_patch_func=_numpy_module_patch,
    )
    bc, k_force, indata_wout = bcovar_payload

    bss_payload = prepare_wout_bss_source_payload(
        state=state,
        static=static,
        indata_wout=indata_wout,
        wout_like=wout_like,
        bc=bc,
        k_force=k_force,
        trig=trig,
        geom=geom,
        lasym=bool(lasym),
        force_sym_func=lambda arr, kind: _force_sym_for_wout(
            arr,
            trig=trig,
            lasym=bool(lasym),
            kind=kind,
        ),
        vmec_forces_rz_from_wout_func=vmec_forces_rz_from_wout,
    )
    (
        use_force_bss, k_force, bsupu_bss, bsupv_bss, ru12_bss, zu12_bss,
        rs_bss, zs_bss, crmn_e_sym, czmn_e_sym, bzmn_e_sym, brmn_e_sym,
        azmn_e_sym, armn_e_sym, geom_bss,
    ) = bss_payload
    _wout_debug_helpers.dump_bsub_parity_if_requested(s=np.asarray(s, dtype=float), bc=bc)
    _wout_debug_helpers.dump_bsubh_if_requested(s=np.asarray(s, dtype=float), bsupu=bsupu_bss, bsupv=bsupv_bss, bc=bc)

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

    pres, presf = pressure_profiles_from_mass_vp(mass=mass, vp=vp, gamma=gamma)

    wint = _vmec_wint_from_trig(trig)
    Aminor_p, Rmajor_p, aspect, volume_p, _ = _compute_aspectratio(
        R=np.asarray(geom["R"]),
        Zu=np.asarray(geom["Zu"]),
        wint=wint,
    )

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
    _wout_debug_helpers.dump_bsub_sources_if_requested(bc=bc)

    # VMEC wrout.f uses the *raw* bsubu/bsubv for Fourier output (bsubumnc/etc).
    # JXBFORCE-style diagnostics (jdotb/Mercier) use the equilibrated + filtered
    # fields. Keep both paths explicit to match VMEC output.
    bsubu_diag = bsubu_out
    bsubv_diag = bsubv_out
    bsub_src = field_options.mercier_bsub_source
    if bsub_src in {"bsubu_e", "bsubu_e_scaled", "bsubu"}:
        u_name = bsub_src
        v_name = bsub_src.replace("bsubu", "bsubv")
        if hasattr(bc, u_name) and hasattr(bc, v_name):
            bsubu_diag = np.asarray(getattr(bc, u_name), dtype=float)
            bsubv_diag = np.asarray(getattr(bc, v_name), dtype=float)
    elif field_options.mercier_use_bsube:
        if hasattr(bc, "bsubu_e") and hasattr(bc, "bsubv_e"):
            bsubu_diag = np.asarray(getattr(bc, "bsubu_e"), dtype=float)
            bsubv_diag = np.asarray(getattr(bc, "bsubv_e"), dtype=float)
    # VMEC fileout.f forces IEQUI=1 before calling funct3d/wrout at output
    # time, regardless of the runtime IEQUI used in iterations.
    iequi = 1
    # VMEC parity: keep the raw bsubv path by default for Mercier/jdotb parity.
    disable_equif_corr = field_options.disable_bsubv_equif_corr
    if (iequi == 1) and (not disable_equif_corr) and getattr(bc, "bsubv_e", None) is not None:
        bsubv_diag = _apply_bsubv_equif_correction(
            bsubv=bsubv_diag,
            bsubv_e=np.asarray(bc.bsubv_e),
            trig=trig,
        )
    t0 = _timing_start(bool(wout_timing_enabled))
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
        apply_scalxc=field_options.apply_bss_scalxc,
    )
    _record_timing(wout_timing, "bsubs_half_s", t0)
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
    skip_bsub_filter = field_options.skip_bsub_filter
    filter_from_raw = field_options.filter_from_raw

    t0 = _timing_start(bool(wout_timing_enabled))
    if bool(lasym):
        use_lasym_loop = field_options.use_lasym_loop
        if (not skip_bsub_filter) and field_options.lasym_filter:
            use_parity_channels = field_options.lasym_filter_use_parity_channels
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

        _wout_debug_helpers.dump_bsub_pre_sym_if_requested(
            trig=trig,
            bsubu=bsubu_out,
            bsubv=bsubv_out,
            bsupu=bsupu_out,
            bsupv=bsupv_out,
            bsubs=bsubs_full,
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
            gmnc, bmnc, bsubumnc, bsubvmnc, bsubsmns, bsupumnc, bsupvmnc = (
                lasym_coeffs[key]
                for key in ("gmnc", "bmnc", "bsubumnc", "bsubvmnc", "bsubsmns", "bsupumnc", "bsupvmnc")
            )
            gmns, bmns, bsubumns, bsubvmns, bsubsmnc, bsupumns, bsupvmns = (
                lasym_coeffs_a[key]
                for key in ("gmns", "bmns", "bsubumns", "bsubvmns", "bsubsmnc", "bsupumns", "bsupvmns")
            )
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

        _zero_first_surface(gmnc, bmnc, bsubumnc, bsubvmnc, bsupumnc, bsupvmnc)
        _zero_first_surface(gmns, bmns, bsubumns, bsubvmns, bsupumns, bsupvmns)
        if (not use_lasym_loop) and (ns > 2):
            bsubsmns[0, :] = 2.0 * bsubsmns[1, :] - bsubsmns[2, :]
            bsubsmnc[0, :] = 2.0 * bsubsmnc[1, :] - bsubsmnc[2, :]
    else:
        use_loop = field_options.symmetric_wrout_loop
        sym_nyq = minimal_wout_symmetric_nyquist_coefficients(
            bc=bc,
            bsubu_out=np.asarray(bsubu_out, dtype=float),
            bsubv_out=np.asarray(bsubv_out, dtype=float),
            bsubs_full=np.asarray(bsubs_full, dtype=float),
            pres=np.asarray(pres, dtype=float),
            ns=int(ns),
            modes=nyq_modes,
            trig=trig,
            use_loop=bool(use_loop),
        )
        (
            gmnc, gmns, bsupumnc, bsupumns, bsupvmnc, bsupvmns, bsubumnc,
            bsubumns, bsubvmnc, bsubvmns, bsubsmns, bsubsmnc, bmnc, bmns,
        ) = sym_nyq
    _record_timing(wout_timing, "nyquist_coeffs_s", t0)
    t0 = _timing_start(bool(wout_timing_enabled))

    # Optional debug path: reconstruct physical real-space fields from the
    # Nyquist coefficients. This is *not* required for the default Mercier/jdotb
    # pipeline, which follows VMEC2000's jxbforce discretization directly on
    # the real-space (bsubu/bsubv) fields.
    bsubu_phys = None
    bsubv_phys = None
    if field_options.mercier_use_wrout_bsubuv:
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
    t_bsub_filter = _timing_start(bool(wout_timing_enabled))
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
            use_bc_parity = field_options.bsub_filter_use_bc_parity
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
            _wout_debug_helpers.dump_bsub_parity_inputs_if_requested(
                bsubu_diag=bsubu_diag,
                bsubv_diag=bsubv_diag,
                bsubu_even=bsubu_even,
                bsubu_odd=bsubu_odd,
                bsubv_even=bsubv_even,
                bsubv_odd=bsubv_odd,
                use_bc_parity=bool(use_bc_parity),
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
    _record_timing(wout_timing, "bsub_filter_s", t_bsub_filter)
    # Match VMEC wrout: bsubu/bsubv Fourier output uses the jxbforce-filtered
    # fields, not the raw bcovar fields.
    bsubu_out = np.asarray(bsubu_diag, dtype=float)
    bsubv_out = np.asarray(bsubv_diag, dtype=float)
    t_bsub_coeffs = _timing_start(bool(wout_timing_enabled))
    if not bool(lasym):
        bsubumnc = _vmec_wrout_nyquist_cos_coeffs(f=bsubu_out, modes=nyq_modes, trig=trig)
        bsubvmnc = _vmec_wrout_nyquist_cos_coeffs(f=bsubv_out, modes=nyq_modes, trig=trig)
        _zero_first_surface(bsubumnc, bsubvmnc)
    _record_timing(wout_timing, "bsub_coeffs_s", t_bsub_coeffs)
    # Keep bsubsmns from the direct bsubs_half computation (wrout.f). The
    # Nyquist-reconstructed path is used only for consistency checks.

    if wout_light:
        buco = np.zeros((ns,), dtype=float)
        bvco = np.zeros((ns,), dtype=float)
        jcuru = np.zeros((ns,), dtype=float)
        jcurv = np.zeros((ns,), dtype=float)
        equif = np.zeros((ns,), dtype=float)
    else:
        t_equif = _timing_start(bool(wout_timing_enabled))
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
        _record_timing(wout_timing, "equif_s", t_equif)

    # Current profile metadata for VMECPlot2.
    current_metadata = _wout_current_profile_metadata_from_indata(indata)

    scalar_diag = compute_minimal_wout_scalar_diagnostics(
        ns=int(ns),
        wout_light=bool(wout_light),
        betatotal=float(betatotal),
        state=state,
        static=static,
        s=np.asarray(s, dtype=float),
        lconm1=bool(getattr(cfg, "lconm1", True)),
        ntor=int(ntor),
        nfp=int(nfp),
        mpol=int(mpol),
        lasym=bool(lasym),
        lbsubs=bool(lbsubs),
        signgs=int(signgs),
        pres=np.asarray(pres, dtype=float),
        vp=np.asarray(vp, dtype=float),
        flux_phips=np.asarray(flux.phips, dtype=float),
        iotas=np.asarray(iotas, dtype=float),
        bc=bc,
        buco=np.asarray(buco, dtype=float),
        trig=trig,
        geom_bss=geom_bss,
        bsupu_bss=np.asarray(bsupu_bss, dtype=float),
        bsupv_bss=np.asarray(bsupv_bss, dtype=float),
        rs_bss=rs_bss,
        zs_bss=zs_bss,
        ru12_bss=ru12_bss,
        zu12_bss=zu12_bss,
        bsubu_diag=np.asarray(bsubu_diag, dtype=float),
        bsubv_diag=np.asarray(bsubv_diag, dtype=float),
        bsubu_raw=np.asarray(bsubu_raw, dtype=float),
        bsubv_raw=np.asarray(bsubv_raw, dtype=float),
        bsubu_phys=bsubu_phys,
        bsubv_phys=bsubv_phys,
        indata=indata,
        timing_enabled=bool(wout_timing_enabled),
        timing=wout_timing,
        vmec_wint_from_trig_func=_vmec_wint_from_trig,
        compute_eqfor_betaxis_func=_compute_eqfor_betaxis,
        compute_eqfor_beta_func=_compute_eqfor_beta,
        compute_ctor_from_buco_func=_compute_ctor_from_buco,
        compute_mercier_func=_compute_mercier,
        glasser_from_wout_mercier_terms_func=_glasser_from_wout_mercier_terms,
    )

    # vmec_jax writes VMEC++-style diagnostic wout files for non-converged runs:
    # preserve every field computed from the last state and mark solver status.
    # Keep an explicit legacy escape hatch for tests/comparisons that require
    # VMEC2000's non-converged beta zeroing behavior.
    zero_beta_nonconv = field_options.zero_nonconverged_beta
    if (not bool(converged)) and bool(zero_beta_nonconv):
        scalar_diag = scalar_diag._replace(betatotal=0.0, betapol=0.0, betator=0.0)

    # Convert internal lambda coefficients to VMEC wout convention.
    from .field import lamscale_from_phips

    lamscale = float(np.asarray(lamscale_from_phips(flux.phips, s)))

    _record_timing(wout_timing, "jxbforce_mercier_s", t0)

    lmns = _lambda_wout_from_full_mesh(
        lam_full=main_geom.lmns_internal,
        m_modes=np.asarray(main_modes.m, dtype=int),
        s=s,
        phipf_internal=phipf_internal,
        lamscale=lamscale,
    )
    lmnc = _lambda_wout_from_full_mesh(
        lam_full=main_geom.lmnc_internal,
        m_modes=np.asarray(main_modes.m, dtype=int),
        s=s,
        phipf_internal=phipf_internal,
        lamscale=lamscale,
    )
    if fsqt is None:
        fsqt_out = np.zeros((100,), dtype=float)
    else:
        fsqt_out = np.asarray(fsqt, dtype=float)

    _wout_debug_helpers.dump_wrout_modes_if_requested(
        ns=int(ns),
        nyq_modes=nyq_modes,
        gmnc=gmnc,
        gmns=gmns,
        bmnc=bmnc,
        bmns=bmns,
        bsubumnc=bsubumnc,
        bsubumns=bsubumns,
        bsubvmnc=bsubvmnc,
        bsubvmns=bsubvmns,
        bsubsmnc=bsubsmnc,
        bsubsmns=bsubsmns,
        bsupumnc=bsupumnc,
        bsupumns=bsupumns,
        bsupvmnc=bsupvmnc,
        bsupvmns=bsupvmns,
    )

    wout = WoutData(**build_minimal_wout_data_kwargs(locals(), path=path, converged=bool(converged)))

    if wout_timing_enabled:
        _wout_debug_helpers.print_wout_timing_if_requested(
            timing=wout_timing,
            total_start=t_wout_total_start,
        )
    return wout


def state_from_wout(wout: WoutData) -> VMECState:
    """Build a :class:`~vmec_jax.state.VMECState` from WOUT Fourier coefficients."""

    from . import field as _field

    return _wout_state_helpers.state_from_wout(
        wout,
        assert_main_modes_match_wout_func=assert_main_modes_match_wout,
        lamscale_from_phips_func=_field.lamscale_from_phips,
    )
