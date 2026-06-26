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
from .state import VMECState
from .fourier import eval_fourier
from .kernels.parity import vmec_m1_internal_to_physical_signed
from .kernels.realspace import (
    vmec_realspace_synthesis,
    vmec_realspace_synthesis_dtheta,
    vmec_realspace_geom_from_state,
)
from .kernels.residue import vmec_pwint_from_trig, vmec_wint_from_trig
from .io.wout_files import bsubs as _wout_bsubs_helpers
from .io.wout_files import debug as _wout_debug_helpers
from .io.wout_files import diagnostics as _wout_diagnostics
from .io.wout_files import flux as _wout_flux_helpers
from .io.wout_files import jxbforce as _wout_jxbforce_helpers
from .io.wout_files import mercier as _wout_mercier
from .io.wout_files import parity as _wout_parity_helpers
from .io.wout_files import state as _wout_state_helpers
from .io.wout_files.minimal import (
    attach_force_payload_geometry,
    build_minimal_wout_data_kwargs,
    compute_minimal_wout_derived_profiles,
    compute_minimal_wout_scalar_diagnostics,
    device_get_if_available,
    env_enabled,
    indata_for_wout_force_path,
    minimal_wout_field_options_from_env,
    minimal_wout_runtime_options_from_env,
    prepare_minimal_wout_core_payload,
    prepare_minimal_wout_force_sources,
    prepare_minimal_wout_nyquist_fields,
)
from .io.wout_files.netcdf import (
    read_wout_payload,
    read_wout_scalar_metadata,
    write_wout_payload,
)
from .io.wout_files.nyquist import (
    apply_nyquist_half_weight as _apply_nyquist_half_weight,  # noqa: F401 - compatibility export
    minimal_wout_lasym_nyquist_coefficients,
    minimal_wout_symmetric_nyquist_coefficients,
    vmec_jxbforce_cos_coeffs as _vmec_jxbforce_cos_coeffs,  # noqa: F401 - compatibility export
    vmec_jxbforce_sin_coeffs as _vmec_jxbforce_sin_coeffs,  # noqa: F401 - compatibility export
    vmec_symforce_antisym as _vmec_symforce_antisym,  # noqa: F401 - compatibility export
    vmec_symforce_apply as _vmec_symforce_apply,
    vmec_symoutput_expand as _vmec_symoutput_expand,
    vmec_symoutput_split as _vmec_symoutput_split,
    vmec_wrout_lasym_bsubuv_output_scale as _vmec_wrout_lasym_bsubuv_output_scale,  # noqa: F401 - compatibility export
    vmec_wrout_nyquist_cos_coeffs as _vmec_wrout_nyquist_cos_coeffs,
    vmec_wrout_nyquist_lasym_loop as _vmec_wrout_nyquist_lasym_loop,  # noqa: F401 - compatibility export
    vmec_wrout_nyquist_sin_coeffs as _vmec_wrout_nyquist_sin_coeffs,  # noqa: F401 - compatibility export
    vmec_wrout_nyquist_sin_coeffs_loop as _vmec_wrout_nyquist_sin_coeffs_loop,  # noqa: F401 - compatibility export
    vmec_wrout_nyquist_synthesis as _vmec_wrout_nyquist_synthesis,  # noqa: F401 - compatibility export
)
from .io.wout_files.schema import (
    WoutData,
    _bool_from_nc,  # noqa: F401 - compatibility export
    _nc_scalar,  # noqa: F401 - compatibility export
    assert_main_modes_match_wout,
)
from .kernels.tomnsp import vmec_trig_tables


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
    return _wout_jxbforce_helpers._solve_jxbforce_collocation(A, rhs)


def _minimal_wout_lasym_nyquist_coefficients_facade(**kwargs):
    """Use facade-level Nyquist hooks for legacy monkeypatch/debug workflows."""

    return minimal_wout_lasym_nyquist_coefficients(
        **kwargs,
        nyquist_cos_coeffs_func=_vmec_wrout_nyquist_cos_coeffs,
        nyquist_sin_coeffs_func=_vmec_wrout_nyquist_sin_coeffs,
    )


def _jxbforce_getbsubs_coeffs_lasym_false(**kwargs) -> np.ndarray | None:
    return _wout_jxbforce_helpers._jxbforce_getbsubs_coeffs_lasym_false(**kwargs)


def _jxbforce_getbsubs_coeffs_lasym_true(**kwargs) -> np.ndarray | None:
    return _wout_jxbforce_helpers._jxbforce_getbsubs_coeffs_lasym_true(**kwargs)


def _jxbforce_apply_bsubs_correction_lasym_false(**kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _wout_jxbforce_helpers._jxbforce_apply_bsubs_correction_lasym_false(
        **kwargs,
        getbsubs_coeffs_func=_jxbforce_getbsubs_coeffs_lasym_false,
    )


def _jxbforce_apply_bsubs_correction_lasym_true(**kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _wout_jxbforce_helpers._jxbforce_apply_bsubs_correction_lasym_true(
        **kwargs,
        getbsubs_coeffs_func=_jxbforce_getbsubs_coeffs_lasym_true,
    )


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
    from .kernels.bcovar import vmec_bcovar_half_mesh_from_wout

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
    from .kernels.numpy_forces import _numpy_module_patch

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
_bsubs_full_mesh_for_wrout = _wout_bsubs_helpers.bsubs_full_mesh_for_wrout
_bsubuv_parity_from_state = _wout_bsubs_helpers.bsubuv_parity_from_state
_bsubuv_parity_from_coeffs = _wout_bsubs_helpers.bsubuv_parity_from_coeffs
_bsubuv_parity_from_realspace_jxbforce = _wout_bsubs_helpers.bsubuv_parity_from_realspace_jxbforce
_bsubuv_parity_from_bcovar = _wout_bsubs_helpers.bsubuv_parity_from_bcovar


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
    from .kernels.bcovar import vmec_bcovar_half_mesh_from_wout
    from .kernels.residue import vmec_force_norms_from_bcovar_dynamic

    runtime_options = minimal_wout_runtime_options_from_env()
    wout_timing_enabled = runtime_options.timing_enabled
    wout_light = runtime_options.light
    wout_fast_bcovar = runtime_options.fast_bcovar
    field_options = minimal_wout_field_options_from_env(wout_light=bool(wout_light))
    wout_timing: dict[str, float] = {}
    t_wout_total_start = _timing_start(bool(wout_timing_enabled))

    core = prepare_minimal_wout_core_payload(
        state=state,
        static=static,
        indata=indata,
        signgs=int(signgs),
        converged=converged,
        flux_override=flux_override,
        profiles_override=profiles_override,
        runtime_options=runtime_options,
        field_options=field_options,
        synthesize_geometry_func=_synthesize_wout_geometry_from_state,
        timing=wout_timing,
        equilibrium_iota_profiles_from_state_func=equilibrium_iota_profiles_from_state,
        chipf_from_chips_func=_chipf_from_chips,
        icurv_full_mesh_from_indata_func=_icurv_full_mesh_from_indata,
    )
    cfg = core.cfg
    ns = core.ns
    mpol = core.mpol
    ntor = core.ntor
    nfp = core.nfp
    lasym = core.lasym
    runtime_options = core.runtime_options
    wout_timing_enabled = core.wout_timing_enabled
    wout_light = core.wout_light
    wout_fast_bcovar = core.wout_fast_bcovar
    field_options = core.field_options
    converged = core.converged
    lbsubs = core.lbsubs
    main_modes = core.main_modes
    nyq_modes = core.nyq_modes
    trig = core.trig
    geom = core.geom
    s = core.s
    flux = core.flux
    chipf_wout = core.chipf_wout
    pres = core.pres
    mass = core.mass
    ncurr = core.ncurr
    iotas = core.iotas
    iotaf = core.iotaf
    gamma = core.gamma
    phipf_internal = core.phipf_internal
    lconm1 = core.lconm1
    main_geom = core.main_geom
    phipf_out = core.phipf_out
    chipf_out = core.chipf_out
    phi = core.phi
    wout_like = core.wout_like
    from .kernels.forces import vmec_forces_rz_from_wout
    from .kernels.numpy_forces import _numpy_module_patch

    force_sources = prepare_minimal_wout_force_sources(
        state=state,
        static=static,
        indata=indata,
        wout_like=wout_like,
        pres=pres,
        geom=geom,
        force_payload_override=force_payload_override,
        fast_bcovar=wout_fast_bcovar,
        timing_enabled=wout_timing_enabled,
        timing=wout_timing,
        lasym=lasym,
        trig=trig,
        vmec_bcovar_half_mesh_from_wout_func=vmec_bcovar_half_mesh_from_wout,
        vmec_forces_rz_from_wout_func=vmec_forces_rz_from_wout,
        numpy_module_patch_func=_numpy_module_patch,
        force_sym_func=lambda arr, kind: _force_sym_for_wout(
            arr,
            trig=trig,
            lasym=bool(lasym),
            kind=kind,
        ),
        dump_bsub_parity_func=lambda *, bc: _wout_debug_helpers.dump_bsub_parity_if_requested(
            s=np.asarray(s, dtype=float),
            bc=bc,
        ),
        dump_bsubh_func=lambda *, bsupu, bsupv, bc: _wout_debug_helpers.dump_bsubh_if_requested(
            s=np.asarray(s, dtype=float),
            bsupu=bsupu,
            bsupv=bsupv,
            bc=bc,
        ),
    )
    (
        bc, k_force, indata_wout, use_force_bss, bsupu_bss, bsupv_bss,
        ru12_bss, zu12_bss, rs_bss, zs_bss, crmn_e_sym, czmn_e_sym,
        bzmn_e_sym, brmn_e_sym, azmn_e_sym, armn_e_sym, geom_bss,
    ) = force_sources

    derived = compute_minimal_wout_derived_profiles(
        bc=bc,
        trig=trig,
        s=s,
        signgs=int(signgs),
        mass=mass,
        gamma=float(gamma),
        geom=geom,
        vmec_force_norms_from_bcovar_dynamic_func=vmec_force_norms_from_bcovar_dynamic,
        vmec_wint_from_trig_func=_vmec_wint_from_trig,
        compute_aspectratio_func=_compute_aspectratio,
    )
    vp = derived.vp
    wb = derived.wb
    wp = derived.wp
    volume = derived.volume
    volume_p = derived.volume_p
    betatotal = derived.betatotal
    pres = derived.pres
    presf = derived.presf
    wint = derived.wint
    Aminor_p = derived.Aminor_p
    Rmajor_p = derived.Rmajor_p
    aspect = derived.aspect

    from .fourier import build_helical_basis
    from .kernels.tomnsp import vmec_angle_grid

    nyquist_fields = prepare_minimal_wout_nyquist_fields(
        state=state,
        static=static,
        cfg=cfg,
        bc=bc,
        k_force=k_force,
        use_force_bss=use_force_bss,
        bsupu_bss=bsupu_bss,
        bsupv_bss=bsupv_bss,
        ru12_bss=ru12_bss,
        zu12_bss=zu12_bss,
        rs_bss=rs_bss,
        zs_bss=zs_bss,
        crmn_e_sym=crmn_e_sym,
        czmn_e_sym=czmn_e_sym,
        geom_bss=geom_bss,
        field_options=field_options,
        trig=trig,
        nyq_modes=nyq_modes,
        pres=pres,
        s=s,
        mpol=int(mpol),
        ntor=int(ntor),
        nfp=int(nfp),
        ns=int(ns),
        lasym=lasym,
        timing_enabled=wout_timing_enabled,
        timing=wout_timing,
        force_sym_func=lambda arr, kind: _force_sym_for_wout(
            arr,
            trig=trig,
            lasym=bool(lasym),
            kind=kind,
        ),
        apply_bsubv_equif_correction_func=_apply_bsubv_equif_correction,
        compute_bsubs_half_mesh_func=_compute_bsubs_half_mesh,
        bsubs_full_mesh_for_wrout_func=_bsubs_full_mesh_for_wrout,
        filter_lasym_loop_func=_filter_bsubuv_jxbforce_lasym_loop,
        filter_symmetric_loop_func=_filter_bsubuv_jxbforce_loop,
        filter_symmetric_parity_func=_filter_bsubuv_jxbforce_parity,
        pshalf_from_s_func=_pshalf_from_s,
        lasym_nyquist_coefficients_func=_minimal_wout_lasym_nyquist_coefficients_facade,
        symmetric_nyquist_coefficients_func=minimal_wout_symmetric_nyquist_coefficients,
        nyquist_cos_coeffs_func=_vmec_wrout_nyquist_cos_coeffs,
        zero_first_surface_func=_zero_first_surface,
        eval_fourier_func=eval_fourier,
        build_helical_basis_func=build_helical_basis,
        vmec_angle_grid_func=vmec_angle_grid,
        dump_bsub_sources_func=_wout_debug_helpers.dump_bsub_sources_if_requested,
        dump_bsub_pre_sym_func=_wout_debug_helpers.dump_bsub_pre_sym_if_requested,
        dump_bsub_parity_inputs_func=_wout_debug_helpers.dump_bsub_parity_inputs_if_requested,
    )
    t0 = _timing_start(bool(wout_timing_enabled))

    if wout_light:
        buco, bvco, jcuru, jcurv, equif = (np.zeros((ns,), dtype=float) for _ in range(5))
    else:
        t_equif = _timing_start(bool(wout_timing_enabled))
        buco, bvco, jcuru, jcurv, equif = _compute_equif_wout(
            bsubu=nyquist_fields.bsubu_out,
            bsubv=nyquist_fields.bsubv_out,
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
        bsubu_diag=np.asarray(nyquist_fields.bsubu_diag, dtype=float),
        bsubv_diag=np.asarray(nyquist_fields.bsubv_diag, dtype=float),
        bsubu_raw=np.asarray(nyquist_fields.bsubu_raw, dtype=float),
        bsubv_raw=np.asarray(nyquist_fields.bsubv_raw, dtype=float),
        bsubu_phys=nyquist_fields.bsubu_phys,
        bsubv_phys=nyquist_fields.bsubv_phys,
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
    fsqt_out = np.zeros((100,), dtype=float) if fsqt is None else np.asarray(fsqt, dtype=float)

    _wout_debug_helpers.dump_wrout_modes_if_requested(
        ns=int(ns),
        nyq_modes=nyq_modes,
        gmnc=nyquist_fields.gmnc,
        gmns=nyquist_fields.gmns,
        bmnc=nyquist_fields.bmnc,
        bmns=nyquist_fields.bmns,
        bsubumnc=nyquist_fields.bsubumnc,
        bsubumns=nyquist_fields.bsubumns,
        bsubvmnc=nyquist_fields.bsubvmnc,
        bsubvmns=nyquist_fields.bsubvmns,
        bsubsmnc=nyquist_fields.bsubsmnc,
        bsubsmns=nyquist_fields.bsubsmns,
        bsupumnc=nyquist_fields.bsupumnc,
        bsupumns=nyquist_fields.bsupumns,
        bsupvmnc=nyquist_fields.bsupvmnc,
        bsupvmns=nyquist_fields.bsupvmns,
    )

    wout_context = dict(locals())
    wout_context.update(nyquist_fields._asdict())
    wout = WoutData(**build_minimal_wout_data_kwargs(wout_context, path=path, converged=bool(converged)))

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
